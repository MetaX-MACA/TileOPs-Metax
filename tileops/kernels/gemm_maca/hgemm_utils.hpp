#ifndef HGEMM_UTILS_HPP_
#define HGEMM_UTILS_HPP_

#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_common.h>
#include <mc_runtime.h>

enum Operation_t { MCBLAS_OP_N, MCBLAS_OP_T };

using FLOAT4 = __NATIVE_VECTOR__(4, float);

#define printf_uint_half2(uintVec, tidIdx)                                                       \
    {                                                                                            \
        if (tid == tidIdx) {                                                                     \
            const T *half2Vec = reinterpret_cast<const T *>(&uintVec);                           \
            printf("%f %f\n", static_cast<float>(half2Vec[0]), static_cast<float>(half2Vec[1])); \
        }                                                                                        \
    }

#define printf_uint2_half4(uint2Vec, tidIdx)                                         \
    {                                                                                \
        if (tid == tidIdx) {                                                         \
            const T *half4Vec = reinterpret_cast<const T *>(&uint2Vec);              \
            printf("%f %f %f %f\n", static_cast<float>(half4Vec[0]),                 \
                   static_cast<float>(half4Vec[1]), static_cast<float>(half4Vec[2]), \
                   static_cast<float>(half4Vec[3]));                                 \
        }                                                                            \
    }

#define printf_uint4_half8(uint8Vec, tidIdx)                            \
    {                                                                   \
        if (tid == tidIdx) {                                            \
            const T *half8Vec = reinterpret_cast<const T *>(&uint8Vec); \
            for (int i = 0; i < 8; ++i) {                               \
                printf("%f ", static_cast<float>(half8Vec[i]));         \
            }                                                           \
            printf("\n");                                               \
        }                                                               \
    }

// #define arrive_gvmcnt(num) __builtin_mxc_arrive(64 + num);
#define arrive_gvmcnt(num) __builtin_mxc_arrive_gvmcnt(num);
#define arrive_bsmcnt(num) __builtin_mxc_arrive_bsmcnt(num);

#define LDG_B128_BSM_NO_PREDICATOR(saddr, gaddr)                                          \
    __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, true, true, false, true, 1, 1, \
                                          MACA_ICMP_EQ);
#define LDG_B128_BSM_WITH_PREDICATOR(saddr, gaddr, cmp_op1, cmp_op2, cmp_type)               \
    __builtin_mxc_ldg_b128_bsm_predicator(saddr, gaddr, 0, true, true, false, true, cmp_op1, \
                                          cmp_op2, cmp_type);
#define LDG_B64_BSM_NO_PREDICATOR(saddr, gaddr)                                          \
    __builtin_mxc_ldg_b64_bsm_predicator(saddr, gaddr, 0, true, true, false, true, 1, 1, \
                                         MACA_ICMP_EQ);
#define LDG_B64_BSM_WITH_PREDICATOR(saddr, gaddr, cmp_op1, cmp_op2, cmp_type)               \
    __builtin_mxc_ldg_b64_bsm_predicator(saddr, gaddr, 0, true, true, false, true, cmp_op1, \
                                         cmp_op2, cmp_type);

template <typename T, bool SwapAB = false>
__forceinline__ __device__ FLOAT4 mma_16x16x16b16(uint a0, uint a1, uint b0, uint b1, FLOAT4 C) {
    using UINT2 = __NATIVE_VECTOR__(2, uint);

    UINT2 A;
    UINT2 B;
    if constexpr (SwapAB) {
        A = UINT2{b0, b1};
        B = UINT2{a0, a1};
    } else {
        A = UINT2{a0, a1};
        B = UINT2{b0, b1};
    }

    if constexpr (std::is_same<T, __half>::value) {
        return __builtin_mxc_mma_16x16x16f16(A, B, C);
    } else {
        return __builtin_mxc_mma_16x16x16bf16(A, B, C);
    }
}

#endif  // HGEMM_UTILS_HPP_
