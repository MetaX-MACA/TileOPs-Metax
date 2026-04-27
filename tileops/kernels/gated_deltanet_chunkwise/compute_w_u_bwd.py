# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

"""
Backward of compute_w_u: given dw, du, compute dk, dv, dbeta (and optionally dAw, dAu).

Forward: w = Aw @ (k*beta), u = Au @ (v*beta) per chunk.
Backward:
  dAw = dw @ (k*beta)^T
  d(k*beta) = Aw^T @ dw   -> dk = d(k*beta) * beta
  dAu = du @ (v*beta)^T
  d(v*beta) = Au^T @ du   -> dv = d(v*beta) * beta
  dbeta = (d(k*beta) * k).sum(-1) + (d(v*beta) * v).sum(-1)

Notes:
  - NOT explicitly materialize (k*beta)/(v*beta). Instead we compute
      dw @ (k*beta)^T  == (dw @ k^T) * beta_col
    i.e. multiply the GEMM result by beta on the column dimension (beta[j]).
  - dk/dv and dbeta are accumulated while looping over tiled K/V dimensions:
      d(k*beta) = Aw^T @ dw,  dbeta += sum_j d(k*beta)[i,j] * k[i,j]
      d(v*beta) = Au^T @ du,  dbeta += sum_j d(v*beta)[i,j] * v[i,j]
"""

import functools

import tilelang
import tilelang.language as T

__all__ = ["compute_w_u_bwd_tl"]


@functools.lru_cache(maxsize=32)
def compute_w_u_bwd_tl(
    batch: int,
    head: int,
    seq_len: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
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
        out_idx=[-5, -4, -3, -2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _kernel_func(num_stages, threads=128):
        @T.macro
        def _body(
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du: T.Tensor([batch, head, seq_len, dim_v], dtype),
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            dAw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            dAu: T.Tensor([batch, head, seq_len, chunk_size], dtype),
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

                # Reused tiles for K-loop and V-loop (reduces shared memory).
                x0 = T.alloc_shared([block_C, tile_d], accum_dtype)
                x1 = T.alloc_shared([block_C, tile_d], accum_dtype)

                dAw_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                dAu_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                acc_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                d_back_frag = T.alloc_fragment([block_C, tile_d], accum_dtype)

                # Load chunk-invariant inputs
                T.copy(Aw[bid, hid, by * block_C : (by + 1) * block_C, :], Aw_s, disable_tma=True)
                T.copy(Au[bid, hid, by * block_C : (by + 1) * block_C, :], Au_s, disable_tma=True)
                T.copy(beta[bid, hid, by * block_C : (by + 1) * block_C], beta_s, disable_tma=True)

                T.clear(dAw_frag)
                T.clear(dAu_frag)
                for i in T.Parallel(block_C):
                    dbeta_s[i] = T.float32(0.0)

                # Loop over K tiles
                for k0 in T.serial(0, sub_dim_k):
                    # x0 := dw_tile, x1 := k_tile
                    T.copy(dw[bid, hid, by * block_C : (by + 1) * block_C, k0 * BK : (k0 + 1) * BK], x0, disable_tma=True)
                    T.copy(k[bid, hid, by * block_C : (by + 1) * block_C, k0 * BK : (k0 + 1) * BK], x1, disable_tma=True)

                    # dAw += dw @ (k*beta)^T
                    #      = (dw @ k^T) * beta_col, where beta_col means scaling column j by beta[j].
                    T.clear(acc_frag)
                    T.gemm(x0, x1, acc_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, block_C):
                        dAw_frag[i, j] += acc_frag[i, j] * beta_s[j]

                    # d_k_beta = Aw^T @ dw  (this is d(k*beta) in math)
                    T.clear(d_back_frag)
                    T.gemm(Aw_s, x0, d_back_frag, transpose_A=True)

                    # dk = d_k_beta * beta_row, dbeta += sum_j d_k_beta[i,j] * k[i,j]
                    for i, j in T.Parallel(block_C, BK):
                        dkb = d_back_frag[i, j]
                        dk[bid, hid, by * block_C + i, k0 * BK + j] = dkb * beta_s[i]
                        x0[i, j] = dkb * x1[i, j]

                    T.reduce_sum(x0, dbeta_tmp, dim=1)
                    for i in T.Parallel(block_C):
                        dbeta_s[i] += dbeta_tmp[i]

                # Loop over V tiles
                for v0 in T.serial(0, sub_dim_v):
                    # x0 := du_tile, x1 := v_tile
                    T.copy(du[bid, hid, by * block_C : (by + 1) * block_C, v0 * BV : (v0 + 1) * BV], x0, disable_tma=True)
                    T.copy(v[bid, hid, by * block_C : (by + 1) * block_C, v0 * BV : (v0 + 1) * BV], x1, disable_tma=True)

                    # dAu += du @ (v*beta)^T
                    #      = (du @ v^T) * beta_col, where beta_col means scaling column j by beta[j].
                    T.clear(acc_frag)
                    T.gemm(x0, x1, acc_frag, transpose_B=True)
                    for i, j in T.Parallel(block_C, block_C):
                        dAu_frag[i, j] += acc_frag[i, j] * beta_s[j]

                    # d_v_beta = Au^T @ du  (this is d(v*beta) in math)
                    T.clear(d_back_frag)
                    T.gemm(Au_s, x0, d_back_frag, transpose_A=True)

                    # dv = d_v_beta * beta_row, dbeta += sum_j d_v_beta[i,j] * v[i,j]
                    for i, j in T.Parallel(block_C, BV):
                        dvb = d_back_frag[i, j]
                        dv[bid, hid, by * block_C + i, v0 * BV + j] = dvb * beta_s[i]
                        x0[i, j] = dvb * x1[i, j]

                    T.reduce_sum(x0, dbeta_tmp, dim=1)
                    for i in T.Parallel(block_C):
                        dbeta_s[i] += dbeta_tmp[i]

                # Write chunk outputs
                T.copy(dAw_frag, dAw[bid, hid, by * block_C : (by + 1) * block_C, :], disable_tma=True)
                T.copy(dAu_frag, dAu[bid, hid, by * block_C : (by + 1) * block_C, :], disable_tma=True)
                for i in T.Parallel(block_C):
                    dbeta[bid, hid, by * block_C + i] = dbeta_s[i]

        @T.prim_func
        def compute_w_u_bwd(
            dw: T.Tensor([batch, head, seq_len, dim_k], dtype),
            du: T.Tensor([batch, head, seq_len, dim_v], dtype),
            Aw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            Au: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            k: T.Tensor([batch, head, seq_len, dim_k], dtype),
            v: T.Tensor([batch, head, seq_len, dim_v], dtype),
            beta: T.Tensor([batch, head, seq_len], dtype),
            dAw: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            dAu: T.Tensor([batch, head, seq_len, chunk_size], dtype),
            dk: T.Tensor([batch, head, seq_len, dim_k], dtype),
            dv: T.Tensor([batch, head, seq_len, dim_v], dtype),
            dbeta: T.Tensor([batch, head, seq_len], dtype),
        ):
            _body(dw, du, Aw, Au, k, v, beta, dAw, dAu, dk, dv, dbeta)

        return compute_w_u_bwd

    return _kernel_func
