"""MACA equal-length Gated DeltaNet prefill.

Skips the Hopper ``fused_gdr_h`` / ``fused_gdr_fwd`` kernels. Those launch
warp-specialized mbarriers, which MACA codegen does not emit. Cumsum and
blocksolve stay in ``prepare.py``. ``w`` / ``u``, the V-tiled state, and the
scaled output live here and do not import the older MACA prefill or forward
modules.
"""

import functools
from typing import Tuple

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention.gdn.prepare import (
    _prefill_blocksolve_A_bthd,
    _prefill_chunk_local_cumsum_bthd_tl,
)

_LOG2E = 1.4426950408889634
_MACA_SMEM_CAP = 65536
_MACA_SMEM_SLACK = 2048
_TILE_CANDIDATES = (64, 32, 16)
_OUTPUT_TILE_CANDIDATES = (128, 64, 32, 16)


def _dtype_nbytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2


def _maca_safe_threads(m: int, n: int, preferred: int, warp_size: int = 64) -> int:
    """Cap threads so MACA MMA can partition gemm ``(M, N)`` across warps."""
    cells = max(1, (m // 16) * (n // 16))
    max_threads = cells * warp_size
    threads = min(int(preferred), int(max_threads))
    threads = max(warp_size, threads - (threads % warp_size))
    return threads


def _require_tile(dim: int, tile: int, axis: str) -> None:
    if tile <= 0:
        raise ValueError(f"{axis} tile must be positive, got {tile}")
    if dim % tile != 0:
        raise ValueError(f"{axis}={dim} is not divisible by tile={tile}")
    if tile % 16 != 0:
        raise ValueError(f"{axis} tile={tile} is not a multiple of 16 (MACA MMA)")


def _h_recurrence_smem(chunk_size: int, dim_k: int, bk: int, bv: int, dtype: str) -> int:
    elem = _dtype_nbytes(dtype)
    return (
        chunk_size * bk * elem * 2
        + chunk_size * 4
        + chunk_size * bv * elem * 2
        + dim_k * bv * elem
        + bk * bv * elem
    )


def _output_o_smem(chunk_size: int, bk: int, bv: int, dtype: str) -> int:
    elem = _dtype_nbytes(dtype)
    return (
        chunk_size * bk * elem * 2
        + chunk_size * 4
        + bk * bv * elem
        + chunk_size * bv * elem
        + chunk_size * chunk_size * elem
    )


def _select_h_block_v(dim_k: int, dim_v: int, streams: int, candidate_bv: int) -> int:
    """Split V when the state grid would otherwise underfill the GPU."""
    programs = int(streams) * (dim_v // candidate_bv)
    min_programs = 64 if dim_k >= 128 and dim_v >= 128 else 32
    if programs < min_programs and candidate_bv > 16 and dim_v % 16 == 0:
        return 16
    return candidate_bv


def _pick_stage_tiles(
    dim_k: int,
    dim_v: int,
    smem_fn,
    stage: str,
    candidates: Tuple[int, ...] = _TILE_CANDIDATES,
) -> Tuple[int, int]:
    best = None
    for bk in candidates:
        if dim_k % bk != 0:
            continue
        for bv in candidates:
            if dim_v % bv != 0:
                continue
            used = smem_fn(bk, bv)
            if used + _MACA_SMEM_SLACK > _MACA_SMEM_CAP:
                continue
            score = (bk * bv, bk + bv)
            if best is None or score > best[0]:
                best = (score, bk, bv)
    if best is None:
        raise ValueError(
            f"MACA {stage}: no BK/BV in {candidates} fits under "
            f"{_MACA_SMEM_CAP} bytes for dim_k={dim_k} dim_v={dim_v}"
        )
    return best[1], best[2]


def _recompute_block_d(dim_k: int, dim_v: int) -> int:
    for tile in _TILE_CANDIDATES:
        if dim_k % tile == 0 and dim_v % tile == 0:
            return tile
    raise ValueError(f"no common MACA tile for dim_k={dim_k} dim_v={dim_v}")


def _plan_dense_prefill_maca(
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    streams: int,
) -> dict:
    if chunk_size % 16 != 0:
        raise ValueError(f"chunk_size={chunk_size} must be a multiple of 16")
    preferred = 256 if chunk_size >= 64 else 128
    h_bk, h_bv = _pick_stage_tiles(
        dim_k,
        dim_v,
        lambda bk, bv: _h_recurrence_smem(chunk_size, dim_k, bk, bv, dtype),
        "h_recurrence",
    )
    h_bv = _select_h_block_v(dim_k, dim_v, streams, h_bv)
    o_bk, o_bv = _pick_stage_tiles(
        dim_k,
        dim_v,
        lambda bk, bv: _output_o_smem(chunk_size, bk, bv, dtype),
        "output_o",
        candidates=_OUTPUT_TILE_CANDIDATES,
    )
    block_d = _recompute_block_d(dim_k, dim_v)
    return {
        "h_block_k": h_bk,
        "h_block_v": h_bv,
        "h_threads": _maca_safe_threads(min(chunk_size, h_bk), h_bv, preferred),
        "o_block_k": o_bk,
        "o_block_v": o_bv,
        "o_threads": min(
            _maca_safe_threads(chunk_size, chunk_size, preferred),
            _maca_safe_threads(chunk_size, o_bv, preferred),
        ),
        "recompute_block_d": block_d,
        "recompute_threads": _maca_safe_threads(chunk_size, block_d, preferred),
    }


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _recompute_w_u_bthd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    block_d: int,
):
    """One block per chunk. Synchronous copy, no mbarrier."""
    block_c = chunk_size
    _require_tile(dim_k, block_d, "dim_k")
    _require_tile(dim_v, block_d, "dim_v")
    num_k_tiles = dim_k // block_d
    num_v_tiles = dim_v // block_d
    accum_dtype = "float32"

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=128):
        @T.prim_func
        def recompute_w_u_bthd(
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            v: T.Tensor([batch, seq_len, head, dim_v], dtype),
            beta: T.Tensor([batch, seq_len, head], dtype),
            A: T.Tensor([batch, seq_len, head, chunk_size], dtype),
            w: T.Tensor([batch, seq_len, head, dim_k], dtype),
            u: T.Tensor([batch, seq_len, head, dim_v], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_c, threads=threads) as (
                bid,
                hid,
                by,
            ):
                base = by * block_c
                A_s = T.alloc_shared([block_c, block_c], dtype)
                beta_s = T.alloc_shared([block_c], dtype)
                x_s = T.alloc_shared([block_c, block_d], dtype)
                out_s = T.alloc_shared([block_c, block_d], dtype)
                out_frag = T.alloc_fragment([block_c, block_d], accum_dtype)

                T.copy(
                    A[bid, base : base + block_c, hid, 0:block_c],
                    A_s,
                    disable_tma=True,
                )
                T.copy(
                    beta[bid, base : base + block_c, hid],
                    beta_s,
                    disable_tma=True,
                )
                T.sync_threads()

                for vt in T.Serial(num_v_tiles):
                    v_offset = vt * block_d
                    T.copy(
                        v[
                            bid,
                            base : base + block_c,
                            hid,
                            v_offset : v_offset + block_d,
                        ],
                        x_s,
                        disable_tma=True,
                    )
                    T.sync_threads()
                    for i, d in T.Parallel(block_c, block_d):
                        x_s[i, d] = x_s[i, d] * beta_s[i]
                    T.clear(out_frag)
                    T.gemm(A_s, x_s, out_frag)
                    T.copy(out_frag, out_s)
                    T.copy(
                        out_s,
                        u[
                            bid,
                            base : base + block_c,
                            hid,
                            v_offset : v_offset + block_d,
                        ],
                        disable_tma=True,
                    )

                for kt in T.Serial(num_k_tiles):
                    k_offset = kt * block_d
                    T.copy(
                        k[
                            bid,
                            base : base + block_c,
                            hid,
                            k_offset : k_offset + block_d,
                        ],
                        x_s,
                        disable_tma=True,
                    )
                    T.sync_threads()
                    for i, d in T.Parallel(block_c, block_d):
                        x_s[i, d] = x_s[i, d] * beta_s[i]
                    T.clear(out_frag)
                    T.gemm(A_s, x_s, out_frag)
                    T.copy(out_frag, out_s)
                    T.copy(
                        out_s,
                        w[
                            bid,
                            base : base + block_c,
                            hid,
                            k_offset : k_offset + block_d,
                        ],
                        disable_tma=True,
                    )

        return recompute_w_u_bthd

    return _func


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _h_recurrence_bhtd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    block_k: int,
    block_v: int,
):
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BK = block_k
    BV = block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(num_stages, threads=128):
        @T.prim_func
        def h_recurrence_bhtd(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
            S_0: T.Tensor([batch, head, dim_k, dim_v], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                k_c = T.alloc_shared([block_C, BK], dtype)
                g_c = T.alloc_shared([block_C], accum_dtype)
                w_c = T.alloc_shared([block_C, BK], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                h_tile = T.alloc_shared([BK, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)

                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                h_next_frag = T.alloc_fragment([BK, BV], accum_dtype)

                v_offset = vid * BV

                T.copy(
                    S_0[bid, hid, :, v_offset : v_offset + BV],
                    h_c,
                    disable_tma=True,
                )
                for i, j in T.Parallel(dim_k, BV):
                    S[bid, hid, 0, i, v_offset + j] = h_c[i, j]

                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(
                        g[bid, hid, t * block_C : (t + 1) * block_C],
                        g_c,
                        disable_tma=True,
                    )
                    T.copy(
                        u[
                            bid,
                            hid,
                            t * block_C : (t + 1) * block_C,
                            v_offset : v_offset + BV,
                        ],
                        u_c,
                        disable_tma=True,
                    )

                    T.clear(ws_frag)
                    for kt in T.serial(0, num_k_tiles):
                        koff = kt * BK
                        T.copy(
                            w[
                                bid,
                                hid,
                                t * block_C : (t + 1) * block_C,
                                koff : koff + BK,
                            ],
                            w_c,
                            disable_tma=True,
                        )
                        for i, j in T.Parallel(BK, BV):
                            h_tile[i, j] = h_c[koff + i, j]
                        T.gemm(w_c, h_tile, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2(
                            (g_c[i] + g_c[block_C - 1]) * _LOG2E
                        )

                    T.copy(
                        v_new_c,
                        v_new[
                            bid,
                            hid,
                            t * block_C : (t + 1) * block_C,
                            v_offset : v_offset + BV,
                        ],
                        disable_tma=True,
                    )

                    for n, j in T.Parallel(block_C, BV):
                        v_new_c[n, j] = v_new_c[n, j] * T.exp2((g_c[block_C - 1] - g_c[n]) * _LOG2E)
                    for kt in T.serial(0, num_k_tiles):
                        koff = kt * BK
                        T.copy(
                            k[
                                bid,
                                hid,
                                t * block_C : (t + 1) * block_C,
                                koff : koff + BK,
                            ],
                            k_c,
                            disable_tma=True,
                        )
                        for i, j in T.Parallel(BK, BV):
                            h_next_frag[i, j] = h_c[koff + i, j] * T.exp2(g_c[block_C - 1] * _LOG2E)
                        T.gemm(
                            k_c,
                            v_new_c,
                            h_next_frag,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(BK, BV):
                            h_c[koff + i, j] = h_next_frag[i, j]
                            S[bid, hid, t + 1, koff + i, v_offset + j] = h_next_frag[i, j]

        return h_recurrence_bhtd

    return _func


@functools.lru_cache(  # lint: target-cache allow: MACA kernel builder is selected by the backend target
    maxsize=32
)
def _output_o_bhtd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    block_k: int,
    block_v: int,
    scale: float,
):
    """Chunk output. ``scale`` multiplies both the QK term and ``q @ h`` in fp32."""
    accum_dtype = "float32"
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BK = block_k
    BV = block_v
    _require_tile(dim_k, BK, "dim_k")
    _require_tile(dim_v, BV, "dim_v")
    num_k_tiles = dim_k // BK
    num_v_tiles = dim_v // BV

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(threads=128):
        @T.prim_func
        def output_o_bhtd(
            q: T.Tensor([batch, head, seq_len, dim_k], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            g: T.Tensor([batch, head, seq_len], dtype),
            S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype),
            v_new: T.Tensor([batch, head, seq_len, dim_v], dtype),
            o: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(num_chunks, num_v_tiles, batch * head, threads=threads) as (
                tid,
                vt,
                bhid,
            ):
                bid = bhid // head
                hid = bhid % head
                base = tid * block_C
                v_offset = vt * BV

                q_c = T.alloc_shared([block_C, BK], dtype)
                k_c = T.alloc_shared([block_C, BK], dtype)
                g_c = T.alloc_shared([block_C], accum_dtype)
                h_c = T.alloc_shared([BK, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)

                o_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)

                T.copy(g[bid, hid, base : base + block_C], g_c, disable_tma=True)
                T.copy(
                    v_new[
                        bid,
                        hid,
                        base : base + block_C,
                        v_offset : v_offset + BV,
                    ],
                    v_new_c,
                    disable_tma=True,
                )

                T.clear(attn_frag)
                for kt in T.serial(0, num_k_tiles):
                    koff = kt * BK
                    T.copy(
                        q[bid, hid, base : base + block_C, koff : koff + BK],
                        q_c,
                        disable_tma=True,
                    )
                    T.copy(
                        k[bid, hid, base : base + block_C, koff : koff + BK],
                        k_c,
                        disable_tma=True,
                    )
                    T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(
                        i >= j,
                        T.cast(
                            attn_frag[i, j] * scale * T.exp2((g_c[i] - g_c[j]) * _LOG2E),
                            dtype,
                        ),
                        T.cast(0, dtype),
                    )

                T.clear(o_frag)
                for kt in T.serial(0, num_k_tiles):
                    koff = kt * BK
                    T.copy(
                        q[bid, hid, base : base + block_C, koff : koff + BK],
                        q_c,
                        disable_tma=True,
                    )
                    T.copy(
                        S[bid, hid, tid, koff : koff + BK, v_offset : v_offset + BV],
                        h_c,
                        disable_tma=True,
                    )
                    T.gemm(q_c, h_c, o_frag)
                for i, j in T.Parallel(block_C, BV):
                    o_frag[i, j] = o_frag[i, j] * scale * T.exp2(g_c[i] * _LOG2E)

                T.gemm(attn, v_new_c, o_frag)
                T.copy(
                    o_frag,
                    o[
                        bid,
                        hid,
                        base : base + block_C,
                        v_offset : v_offset + BV,
                    ],
                    disable_tma=True,
                )

        return output_o_bhtd

    return _func


def run_dense_prefill_maca(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    *,
    scale: float = 1.0,
) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor]:
    """BTHD prefill. Returns ``(o, None, final_state, g_cum)``."""
    batch, seq_len, head, dim_k = q.shape
    dim_v = v.shape[-1]
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len={seq_len} is not divisible by chunk_size={chunk_size}")
    dtype = str(q.dtype).split(".")[-1]
    scale = float(scale)
    plan = _plan_dense_prefill_maca(chunk_size, dim_k, dim_v, dtype, batch * head)

    g_cum = _prefill_chunk_local_cumsum_bthd_tl(batch, head, seq_len, chunk_size, dtype)(g)
    # Decay stays inside A. The CUDA fused kernel applies it later and passes
    # use_gate=False; this recurrence expects that decay in w and u.
    inverse = _prefill_blocksolve_A_bthd(k, g_cum, beta, chunk_size, use_gate=True)
    w, u = _recompute_w_u_bthd_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        plan["recompute_block_d"],
    )(plan["recompute_threads"])(k, v, beta, inverse)

    q_h = q.permute(0, 2, 1, 3).contiguous()
    k_h = k.permute(0, 2, 1, 3).contiguous()
    g_h = g_cum.permute(0, 2, 1).contiguous()
    w_h = w.permute(0, 2, 1, 3).contiguous()
    u_h = u.permute(0, 2, 1, 3).contiguous()
    s0 = torch.zeros(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    states, v_new = _h_recurrence_bhtd_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        plan["h_block_k"],
        plan["h_block_v"],
    )(0, plan["h_threads"])(k_h, g_h, w_h, u_h, s0)
    o_h = _output_o_bhtd_tl(
        batch,
        head,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        dtype,
        plan["o_block_k"],
        plan["o_block_v"],
        scale,
    )(plan["o_threads"])(q_h, k_h, g_h, states, v_new)
    o = o_h.permute(0, 2, 1, 3).contiguous()
    final_state = states[:, :, -1, :, :].contiguous().to(torch.float32)
    return o, None, final_state, g_cum


class GatedDeltaNetDensePrefillFwdMACAKernel(Kernel):
    """MACA equal-length BTHD inference prefill.

    Same call signature as the Hopper kernel. Cumsum and blocksolve stay
    shared; partition and ``fused_gdr_*`` are replaced by the V-tiled recurrence.
    """

    supported_archs = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        scale: float,
        dtype: torch.dtype,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.scale = scale
        self.dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del initial_state, cu_seqlens, cu_seqlens_cpu, A_log, dt_bias
        self._require_cuda(q=q, k=k, v=v, g=g, beta=beta)
        o, _states, final_state, _g_cum = run_dense_prefill_maca(
            q,
            k,
            v,
            g,
            beta,
            chunk_size=64,
            scale=self.scale,
        )
        return o, final_state
