#pragma once

#ifndef TILEOPS_MACA_HGEMM_EXAMPLE_HEADER
#define TILEOPS_MACA_HGEMM_EXAMPLE_HEADER "maca_hgemm_example_tuned.hpp"
#endif

using INT128 = __NATIVE_VECTOR__(4, int);

#ifndef TILEOPS_MACA_HGEMM_USE_ASYNC_A_STAGING
#define TILEOPS_MACA_HGEMM_USE_ASYNC_A_STAGING 0
#endif
#ifndef TILEOPS_MACA_HGEMM_USE_EXAMPLE_ASYNC_A_STAGING
#define TILEOPS_MACA_HGEMM_USE_EXAMPLE_ASYNC_A_STAGING 0
#endif

#if TILEOPS_MACA_HGEMM_USE_EXAMPLE_ASYNC_A_STAGING || TILEOPS_MACA_HGEMM_USE_ASYNC_A_STAGING
#define TILEOPS_MACA_A_LDG(dst, src, use_predicate, cmp_limit, cmp_op)          \
    do {                                                                         \
        (void)(use_predicate);                                                   \
        (void)(cmp_limit);                                                       \
        (void)(cmp_op);                                                          \
        *reinterpret_cast<ALdgType *>(dst) = __builtin_mxc_load_global_async128( \
            reinterpret_cast<INT128 *>(src));                                    \
    } while (0)
#else
#define TILEOPS_MACA_A_LDG(dst, src, use_predicate, cmp_limit, cmp_op)          \
    do {                                                                         \
        if (use_predicate) {                                                     \
            __builtin_mxc_ldg_b128_bsm_predicator(                               \
                dst, src, 0, true, true, false, true, 0, cmp_limit, cmp_op);     \
        } else {                                                                 \
            LDG_B128_BSM_NO_PREDICATOR(dst, src);                                \
        }                                                                        \
    } while (0)
#endif

#include TILEOPS_MACA_HGEMM_EXAMPLE_HEADER
#include "maca_hgemm_layout_b.hpp"

__forceinline__ __device__ void tileops_maca_hgemm_tn_f16_beta0(
    const void *a,
    const void *b,
    void *c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int bidx,
    int bidy) {
    hgemm_tn_128x128x128_4m1n8k_256t_device<__half, __half, float, true>(
        a, b, c, m, n, k, lda, ldb, ldc, 1.0f, 0.0f, bidx, bidy);
}

__forceinline__ __device__ void tileops_maca_hgemm_tn_f16_to_f32_beta0(
    const void *a,
    const void *b,
    void *c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int bidx,
    int bidy) {
    hgemm_tn_128x128x128_4m1n8k_256t_device<__half, float, float, true>(
        a, b, c, m, n, k, lda, ldb, ldc, 1.0f, 0.0f, bidx, bidy);
}

__forceinline__ __device__ void tileops_maca_hgemm_layout_ab_tn_f16_beta0(
    const void *a,
    const void *b,
    void *c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int bidx,
    int bidy) {
    muxi_layout_kernels::layout_hgemm_tn_128x128x128_4m1n8k_256t_device<
        __half, __half, float, true, false>(
        a, b, c, m, n, k, lda, ldb, ldc, 1.0f, 0.0f, nullptr, bidx, bidy);
}

__forceinline__ __device__ void tileops_maca_hgemm_tn_f16_splitk_atomic_beta0(
    const void *a,
    const void *b,
    void *c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int bidx,
    int bidy) {
    hgemm_tn_128x128x128_4m1n8k_256t_device<__half, __half, float, true, true>(
        a, b, c, m, n, k, lda, ldb, ldc, 1.0f, 0.0f, bidx, bidy);
}

__forceinline__ __device__ void tileops_maca_hgemm_tn_f32_splitk_atomic_beta0(
    const void *a,
    const void *b,
    void *c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int bidx,
    int bidy) {
    hgemm_tn_128x128x128_4m1n8k_256t_device<__half, float, float, true, true>(
        a, b, c, m, n, k, lda, ldb, ldc, 1.0f, 0.0f, bidx, bidy);
}

__forceinline__ __device__ void tileops_maca_hgemm_splitk_reduce_f32_to_f16(
    const void *partial_c,
    void *c,
    int m,
    int n,
    int split_k,
    int total,
    int flat_idx) {
    if (flat_idx >= total) {
        return;
    }

    const float *partial = static_cast<const float *>(partial_c);
    __half *out = static_cast<__half *>(c);

    float acc = 0.0f;
    for (int sk = 0; sk < split_k; ++sk) {
        acc += partial[sk * n * m + flat_idx];
    }
    out[flat_idx] = static_cast<__half>(acc);
}

__forceinline__ __device__ void tileops_maca_hgemm_cast_f32_to_f16(
    const void *src,
    void *dst,
    int total,
    int flat_idx) {
    if (flat_idx >= total) {
        return;
    }

    const float *in = static_cast<const float *>(src);
    __half *out = static_cast<__half *>(dst);
    out[flat_idx] = static_cast<__half>(in[flat_idx]);
}


__forceinline__ __device__ void tileops_maca_hgemm_splitk_reduce_f16_to_f16(
    const void *partial_c,
    void *c,
    int m,
    int n,
    int split_k,
    int total,
    int flat_idx) {
    if (flat_idx >= total) {
        return;
    }

    const __half *partial = static_cast<const __half *>(partial_c);
    __half *out = static_cast<__half *>(c);

    float acc = 0.0f;
    for (int sk = 0; sk < split_k; ++sk) {
        acc += static_cast<float>(partial[sk * n * m + flat_idx]);
    }
    out[flat_idx] = static_cast<__half>(acc);
}
