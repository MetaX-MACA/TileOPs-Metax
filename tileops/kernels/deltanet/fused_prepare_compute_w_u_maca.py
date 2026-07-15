"""
Low-shared-memory fused prepare_wy_repr + compute_w_u kernel for
ungated DeltaNet.

Pipeline per chunk:
  1. Compute K @ K^T with a tiled K dimension.
  2. Build P = -strictLower(diag(beta) @ (K @ K^T)).
  3. Compute A^{-1} with the Neumann-series doubling method.
  4. Write A^{-1} to Aw and Au.
  5. Reuse P_shared as the RHS workspace for w and u GEMMs.
"""

import functools
import math

import tilelang
import tilelang.language as T

__all__ = ["fused_prepare_compute_w_u_tl_maca"]


def _select_block_k(dim_k: int) -> int:
    """Choose a small tensor-core-friendly K tile that divides dim_k."""
    if dim_k % 32 == 0:
        return 32
    if dim_k % 16 == 0:
        return 16
    return dim_k


@functools.lru_cache(maxsize=32)
def fused_prepare_compute_w_u_tl_maca(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """Fused TileLang kernel: (k, v, beta) -> (Aw, Au, w, u)."""
    if seq_len % chunk_size != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by chunk_size ({chunk_size})"
        )

    accum_dtype = "float32"
    block_C = chunk_size
    block_K = _select_block_k(dim_k)
    num_k_tiles = dim_k // block_K

    # P_shared is [BC, BC]. For w/u, use the first block_R columns as
    # one RHS tile and zero-fill the remaining columns.
    block_R = math.gcd(block_C, math.gcd(dim_k, dim_v))
    num_w_tiles = dim_k // block_R
    num_u_tiles = dim_v // block_R

    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0

    @tilelang.jit(
        out_idx=[-4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _fused_func(num_stages, threads=128):
        # num_stages is retained for API/config compatibility. The K loop
        # deliberately uses one explicit shared tile; the parameter remains
        # unused so pipeline multibuffering cannot increase shared memory.

        @T.macro
        def _fused_body(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            with T.Kernel(
                batch,
                head,
                seq_len // block_C,
                threads=threads,
            ) as (bid, hid, by):
                chunk_offset = by * block_C

                # Shared memory:
                #   k_tile_shared is reused as beta cache after Gram finishes.
                #   P_shared is reused as the RHS workspace after inversion.
                k_tile_shared = T.alloc_shared([block_C, block_K], dtype)
                S_shared = T.alloc_shared([block_C, block_C], accum_dtype)
                P_shared = T.alloc_shared([block_C, block_C], accum_dtype)

                # One fp32 fragment is reused for Gram, Neumann products,
                # and the w/u output tiles.
                mat_frag = T.alloc_fragment([block_C, block_C], accum_dtype)

                # ----------------------------------------------------------
                # 1. Gram = k @ k^T, tiled over dim_k.
                # ----------------------------------------------------------
                T.clear(mat_frag)
                for tk in T.Serial(num_k_tiles):
                    k_offset = tk * block_K
                    T.copy(
                        k[
                            bid,
                            hid,
                            chunk_offset : chunk_offset + block_C,
                            k_offset : k_offset + block_K,
                        ],
                        k_tile_shared,
                        disable_tma=True,
                    )
                    T.sync_threads()

                    # T.gemm accumulates into mat_frag; it was cleared once
                    # before entering the K-tile loop.
                    T.gemm(
                        k_tile_shared,
                        k_tile_shared,
                        mat_frag,
                        transpose_B=True,
                    )

                    # Protect the shared tile before the next iteration
                    # overwrites it.
                    T.sync_threads()

                # The K tile is dead now. Reuse its first column to cache beta.
                for i in T.Parallel(block_C):
                    k_tile_shared[i, 0] = beta[bid, hid, chunk_offset + i]
                T.sync_threads()

                # ----------------------------------------------------------
                # 2. P = -strictLower(diag(beta) * Gram), S = I.
                # ----------------------------------------------------------
                for i, j in T.Parallel(block_C, block_C):
                    P_shared[i, j] = T.if_then_else(
                        i > j,
                        -mat_frag[i, j]
                        * T.cast(k_tile_shared[i, 0], accum_dtype),
                        T.float32(0.0),
                    )
                    S_shared[i, j] = T.if_then_else(
                        i == j,
                        T.float32(1.0),
                        T.float32(0.0),
                    )
                T.sync_threads()

                # ----------------------------------------------------------
                # 3. S = I + P + P^2 + ... using repeated squaring.
                # ----------------------------------------------------------
                for _r in T.Serial(num_rounds):
                    T.clear(mat_frag)
                    T.gemm(P_shared, S_shared, mat_frag)

                    for i, j in T.Parallel(block_C, block_C):
                        S_shared[i, j] = S_shared[i, j] + mat_frag[i, j]

                    T.clear(mat_frag)
                    T.gemm(P_shared, P_shared, mat_frag)
                    T.copy(mat_frag, P_shared)

                    # Both S_shared and P_shared are consumed in the next
                    # iteration, so make the phase boundary explicit.
                    T.sync_threads()

                # ----------------------------------------------------------
                # 4. Write A^{-1} to Aw and Au.
                # ----------------------------------------------------------
                T.copy(S_shared, mat_frag)
                T.copy(
                    mat_frag,
                    Aw[
                        bid,
                        hid,
                        chunk_offset : chunk_offset + block_C,
                        :,
                    ],
                    disable_tma=True,
                )
                T.copy(
                    mat_frag,
                    Au[
                        bid,
                        hid,
                        chunk_offset : chunk_offset + block_C,
                        :,
                    ],
                    disable_tma=True,
                )
                T.sync_threads()

                # ----------------------------------------------------------
                # 5. w = S @ (k * beta).
                #    Reuse P_shared as an fp32 RHS tile.
                # ----------------------------------------------------------
                for tw in T.Serial(num_w_tiles):
                    rhs_offset = tw * block_R

                    for i, j in T.Parallel(block_C, block_C):
                        P_shared[i, j] = T.float32(0.0)
                    for i, j in T.Parallel(block_C, block_R):
                        P_shared[i, j] = (
                            T.cast(
                                k[
                                    bid,
                                    hid,
                                    chunk_offset + i,
                                    rhs_offset + j,
                                ],
                                accum_dtype,
                            )
                            * T.cast(k_tile_shared[i, 0], accum_dtype)
                        )
                    T.sync_threads()

                    T.clear(mat_frag)
                    T.gemm(S_shared, P_shared, mat_frag)

                    for i, j in T.Parallel(block_C, block_R):
                        w[
                            bid,
                            hid,
                            chunk_offset + i,
                            rhs_offset + j,
                        ] = T.cast(mat_frag[i, j], dtype)

                    # Ensure every GEMM reader has finished before P_shared is
                    # overwritten by the next RHS tile.
                    T.sync_threads()

                # ----------------------------------------------------------
                # 6. u = S @ (v * beta), using the same RHS workspace.
                # ----------------------------------------------------------
                for tu in T.Serial(num_u_tiles):
                    rhs_offset = tu * block_R

                    for i, j in T.Parallel(block_C, block_C):
                        P_shared[i, j] = T.float32(0.0)
                    for i, j in T.Parallel(block_C, block_R):
                        P_shared[i, j] = (
                            T.cast(
                                v[
                                    bid,
                                    hid,
                                    chunk_offset + i,
                                    rhs_offset + j,
                                ],
                                accum_dtype,
                            )
                            * T.cast(k_tile_shared[i, 0], accum_dtype)
                        )
                    T.sync_threads()

                    T.clear(mat_frag)
                    T.gemm(S_shared, P_shared, mat_frag)

                    for i, j in T.Parallel(block_C, block_R):
                        u[
                            bid,
                            hid,
                            chunk_offset + i,
                            rhs_offset + j,
                        ] = T.cast(mat_frag[i, j], dtype)

                    T.sync_threads()

        @T.prim_func
        def fused_prepare_compute_w_u(
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            w: T.Tensor([batch, head, seq_len, dim_k], dtype),
            u: T.Tensor([batch, head, seq_len, dim_v], dtype),
        ):
            _fused_body(k, v, beta, Aw, Au, w, u)

        return fused_prepare_compute_w_u

    return _fused_func