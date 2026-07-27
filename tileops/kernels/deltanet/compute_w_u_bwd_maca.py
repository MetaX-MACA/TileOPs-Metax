"""
Tiled backward of compute_w_u + A_inv backward (MACA smem-safe path).

wu_bwd uses K/V tiling (BK=BV=32) to keep shared memory under 64 KiB.
dw_corr, du merge, and dk merge with dk_partial/dk_corr run in Python
(see deltanet_bwd_maca._deltanet_bwd_wrapped_kernel_maca).
"""

import functools

import tilelang
import tilelang.language as T

__all__ = ["compute_w_u_bwd_tl_maca"]


@functools.lru_cache(maxsize=32)
def compute_w_u_bwd_tl_maca(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """TileLang: tiled wu_bwd + A_inv backward; outputs dk_wu, dv, dbeta."""
    accum_dtype = "float32"
    block_C = chunk_size

    BK = 32
    BV = 32
    tile_d = BK
    sub_dim_k = dim_k // BK
    sub_dim_v = dim_v // BV

    assert dim_k % BK == 0, "dim_k must be divisible by BK"
    assert dim_v % BV == 0, "dim_v must be divisible by BV"

    @tilelang.jit(
        out_idx=[-3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _kernel_func(num_stages, threads=128):
        @T.prim_func
        def compute_w_u_bwd_maca(
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du: T.Tensor([batch, head, seq_len, dim_v], dtype),
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dv: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dbeta: T.Tensor([batch, head, seq_len], dtype),
        ):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, by):
                Aw_s = T.alloc_shared([block_C, block_C], accum_dtype)
                Au_s = T.alloc_shared([block_C, block_C], accum_dtype)
                beta_s = T.alloc_shared([block_C], accum_dtype)
                dbeta_s = T.alloc_shared([block_C], accum_dtype)
                dbeta_tmp = T.alloc_shared([block_C], accum_dtype)
                x0 = T.alloc_shared([block_C, tile_d], accum_dtype)
                x1 = T.alloc_shared([block_C, tile_d], accum_dtype)
                work_bc = T.alloc_shared([block_C, block_C], accum_dtype)

                dAw_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                dAu_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                acc_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_back_frag = T.alloc_fragment([block_C, tile_d], accum_dtype)
                dA_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                kkt_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                dk_A_frag = T.alloc_fragment([block_C, tile_d], accum_dtype)

                T.copy(Aw[bid, hid, by * block_C : (by + 1) * block_C, :], Aw_s, disable_tma=True)
                T.copy(Au[bid, hid, by * block_C : (by + 1) * block_C, :], Au_s, disable_tma=True)
                T.copy(beta[bid, hid, by * block_C : (by + 1) * block_C], beta_s, disable_tma=True)

                T.clear(dAw_frag)
                T.clear(dAu_frag)
                for i in T.Parallel(block_C):
                    dbeta_s[i] = T.float32(0.0)

                for k0 in T.serial(0, sub_dim_k):
                    T.copy(
                        dw[bid, hid, by * block_C : (by + 1) * block_C, k0 * BK : (k0 + 1) * BK],
                        x0, disable_tma=True,
                    )
                    T.copy(
                        k[bid, hid, by * block_C : (by + 1) * block_C, k0 * BK : (k0 + 1) * BK],
                        x1, disable_tma=True,
                    )

                    T.clear(acc_frag)
                    T.gemm(x0, x1, acc_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, block_C):
                        dAw_frag[i, j] += acc_frag[i, j] * beta_s[j]

                    T.clear(d_back_frag)
                    T.gemm(Aw_s, x0, d_back_frag, transpose_A=True)

                    for i, j in T.Parallel(block_C, BK):
                        dkb = d_back_frag[i, j]
                        dk[bid, hid, by * block_C + i, k0 * BK + j] = dkb * beta_s[i]
                        x0[i, j] = dkb * x1[i, j]

                    T.reduce_sum(x0, dbeta_tmp, dim=1)
                    for i in T.Parallel(block_C):
                        dbeta_s[i] += dbeta_tmp[i]

                for v0 in T.serial(0, sub_dim_v):
                    T.copy(
                        du[bid, hid, by * block_C : (by + 1) * block_C, v0 * BV : (v0 + 1) * BV],
                        x0, disable_tma=True,
                    )
                    T.copy(
                        v[bid, hid, by * block_C : (by + 1) * block_C, v0 * BV : (v0 + 1) * BV],
                        x1, disable_tma=True,
                    )

                    T.clear(acc_frag)
                    T.gemm(x0, x1, acc_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, block_C):
                        dAu_frag[i, j] += acc_frag[i, j] * beta_s[j]

                    T.clear(d_back_frag)
                    T.gemm(Au_s, x0, d_back_frag, transpose_A=True)

                    for i, j in T.Parallel(block_C, BV):
                        dvb = d_back_frag[i, j]
                        dv[bid, hid, by * block_C + i, v0 * BV + j] = dvb * beta_s[i]
                        x0[i, j] = dvb * x1[i, j]

                    T.reduce_sum(x0, dbeta_tmp, dim=1)
                    for i in T.Parallel(block_C):
                        dbeta_s[i] += dbeta_tmp[i]

                for i, j in T.Parallel(block_C, block_C):
                    work_bc[i, j] = dAw_frag[i, j] + dAu_frag[i, j]

                T.clear(dA_frag)
                T.gemm(work_bc, Aw_s, dA_frag, transpose_B=True)
                T.copy(dA_frag, work_bc)
                T.clear(dA_frag)
                T.gemm(Aw_s, work_bc, dA_frag, transpose_A=True)

                for i, j in T.Parallel(block_C, block_C):
                    work_bc[i, j] = T.if_then_else(i > j, -dA_frag[i, j], T.float32(0.0))

                for k0 in T.serial(0, sub_dim_k):
                    T.copy(
                        k[bid, hid, by * block_C : (by + 1) * block_C, k0 * BK : (k0 + 1) * BK],
                        x1, disable_tma=True,
                    )
                    for i, j in T.Parallel(block_C, BK):
                        x0[i, j] = x1[i, j] * beta_s[i]

                    T.clear(dk_A_frag)
                    T.gemm(work_bc, x1, dk_A_frag)
                    for i, j in T.Parallel(block_C, BK):
                        dk_A_frag[i, j] = dk_A_frag[i, j] * beta_s[i]

                    T.clear(d_back_frag)
                    T.gemm(work_bc, x0, d_back_frag, transpose_A=True)
                    for i, j in T.Parallel(block_C, BK):
                        dk[bid, hid, by * block_C + i, k0 * BK + j] += (
                            dk_A_frag[i, j] + d_back_frag[i, j]
                        )

                T.clear(kkt_frag)
                for k0 in T.serial(0, sub_dim_k):
                    T.copy(
                        k[bid, hid, by * block_C : (by + 1) * block_C, k0 * BK : (k0 + 1) * BK],
                        x1, disable_tma=True,
                    )
                    T.clear(acc_frag)
                    T.gemm(x1, x1, acc_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, block_C):
                        kkt_frag[i, j] += acc_frag[i, j]

                for i, j in T.Parallel(block_C, block_C):
                    work_bc[i, j] = work_bc[i, j] * kkt_frag[i, j]
                T.reduce_sum(work_bc, dbeta_tmp, dim=1)
                for i in T.Parallel(block_C):
                    dbeta[bid, hid, by * block_C + i] = dbeta_s[i] + dbeta_tmp[i]

        return compute_w_u_bwd_maca

    return _kernel_func
