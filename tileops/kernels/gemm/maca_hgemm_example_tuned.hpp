#include <cmath>
#include <type_traits>
#include "hgemm_utils.hpp"

#ifndef TILEOPS_MACA_A_LDG
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

template <typename T, typename Tc, typename Tscal, bool IsBetaZero, bool UseAtomicAdd = false>
__forceinline__ __device__ void hgemm_tn_128x128x128_4m1n8k_256t_device(const void *A,
                                                                        const void *B,
                                                                        void *C,
                                                                        int M,
                                                                        int N,
                                                                        int K,
                                                                        int lda,
                                                                        int ldb,
                                                                        int ldc,
                                                                        Tscal alpha,
                                                                        Tscal beta,
                                                                        int bidx,
                                                                        int bidy) {
    constexpr int TileM = 128;
    constexpr int TileN = 128;
    constexpr int Stage = 4;

    using ALdgType = __NATIVE_VECTOR__(4, uint);
    using BLdgType = __NATIVE_VECTOR__(4, uint);
    using CStgType = __NATIVE_VECTOR__(sizeof(Tc), uint);  // 4 FP32, or 4 FP16, or 4 BF16
    using ALdsType = ALdgType;
    using BLdsType = BLdgType;
    using FLOAT4 = __NATIVE_VECTOR__(4, float);

    uint8_t *APtr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(A));
    uint8_t *BPtr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(B));
    CStgType *CPtr = reinterpret_cast<CStgType *>(C);

    lda /= sizeof(ALdgType) / sizeof(T);   // lda needs to be align to 8
    ldb /= sizeof(BLdgType) / sizeof(T);   // ldb needs to be align to 8
    ldc /= sizeof(CStgType) / sizeof(Tc);  // ldc needs to be align to 4

    const int startRow = bidx * TileM;
    const int startCol = bidy * TileN;

    M -= startRow;
    N -= startCol;

    const int M_A = M;
    const int N_B = N;
    const int M_C = M / (sizeof(CStgType) / sizeof(Tc));
    const int N_C = N;

    APtr += startRow * lda * sizeof(ALdgType);
    // BPtr += startCol * ldb;
    BPtr += startCol * ldb * sizeof(BLdgType);
    CPtr += startCol * ldc + startRow / (sizeof(CStgType) / sizeof(Tc));

    const int tid = threadIdx.x;
    // __builtin_mxc_readfirstlane可强制让slot为streg
    const int slot = __builtin_mxc_readfirstlane(tid / 64);
    const int lane = tid & 63;

    const int A_row = (tid / 16) * 4;
    const int A_col = (tid & 15) ^ (tid / 16);
    const int B_col = A_row;
    const int B_row = A_col;

    int ALdgOffset[2][4];
    ALdgOffset[0][0] = (A_col + lda * (A_row + 0 < M_A ? A_row + 0 : M_A - 1)) * sizeof(ALdgType);
    ALdgOffset[0][1] = (A_col + lda * (A_row + 1 < M_A ? A_row + 1 : M_A - 1)) * sizeof(ALdgType);
    ALdgOffset[0][2] = (A_col + lda * (A_row + 2 < M_A ? A_row + 2 : M_A - 1)) * sizeof(ALdgType);
    ALdgOffset[0][3] = (A_col + lda * (A_row + 3 < M_A ? A_row + 3 : M_A - 1)) * sizeof(ALdgType);
    ALdgOffset[1][0] =
        (A_col + lda * (A_row + 64 + 0 < M_A ? A_row + 64 + 0 : M_A - 1)) * sizeof(ALdgType);
    ALdgOffset[1][1] =
        (A_col + lda * (A_row + 64 + 1 < M_A ? A_row + 64 + 1 : M_A - 1)) * sizeof(ALdgType);
    ALdgOffset[1][2] =
        (A_col + lda * (A_row + 64 + 2 < M_A ? A_row + 64 + 2 : M_A - 1)) * sizeof(ALdgType);
    ALdgOffset[1][3] =
        (A_col + lda * (A_row + 64 + 3 < M_A ? A_row + 64 + 3 : M_A - 1)) * sizeof(ALdgType);
    int BLdgOffset[2][4];
    BLdgOffset[0][0] = (B_row + ldb * (B_col + 0 < N_B ? B_col + 0 : N_B - 1)) * sizeof(BLdgType);
    BLdgOffset[0][1] = (B_row + ldb * (B_col + 1 < N_B ? B_col + 1 : N_B - 1)) * sizeof(BLdgType);
    BLdgOffset[0][2] = (B_row + ldb * (B_col + 2 < N_B ? B_col + 2 : N_B - 1)) * sizeof(BLdgType);
    BLdgOffset[0][3] = (B_row + ldb * (B_col + 3 < N_B ? B_col + 3 : N_B - 1)) * sizeof(BLdgType);
    BLdgOffset[1][0] =
        (B_row + ldb * (B_col + 64 + 0 < N_B ? B_col + 64 + 0 : N_B - 1)) * sizeof(BLdgType);
    BLdgOffset[1][1] =
        (B_row + ldb * (B_col + 64 + 1 < N_B ? B_col + 64 + 1 : N_B - 1)) * sizeof(BLdgType);
    BLdgOffset[1][2] =
        (B_row + ldb * (B_col + 64 + 2 < N_B ? B_col + 64 + 2 : N_B - 1)) * sizeof(BLdgType);
    BLdgOffset[1][3] =
        (B_row + ldb * (B_col + 64 + 3 < N_B ? B_col + 64 + 3 : N_B - 1)) * sizeof(BLdgType);

    int lds_row = tid & 15;
    int lds_col[4];
    for (int i = 0; i < 4; ++i) {
        lds_col[i] = (4 * i + lane / 16) ^ lds_row;
    }

    int ALdsOffset[4];
    int BLdsOffset[4];
    const int ALdsOffsetTmp = (slot / 2) * 256;
    const int BLdsOffsetTmp = (slot & 1) * 256 + (0x2000 / sizeof(ALdsType));
    for (int i = 0; i < 4; ++i) {
        const int tmp = lds_row * 16 + lds_col[i];
        BLdsOffset[i] = (tmp + BLdsOffsetTmp) * sizeof(ALdsType);
        ALdsOffset[i] = (tmp + ALdsOffsetTmp) * sizeof(ALdsType);
    }

    __shared__ uint8_t WSM[0x10000];  // 64KB

    FLOAT4 C_f32[4][4] = {0};
    ALdsType a[4][4];
    BLdsType b[4][4];

    uint8_t *WSM_Ldg = WSM + slot * 0x400;

    for (int stage_i = 0; stage_i < Stage; ++stage_i) {
        TILEOPS_MACA_A_LDG(
            WSM_Ldg + 0x4000 * stage_i + 0x0000, APtr + ALdgOffset[0][stage_i], true,
            K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
        TILEOPS_MACA_A_LDG(
            WSM_Ldg + 0x4000 * stage_i + 0x1000, APtr + ALdgOffset[1][stage_i], true,
            K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
        __builtin_mxc_ldg_b128_bsm_predicator(
            WSM_Ldg + 0x4000 * stage_i + 0x2000, BPtr + BLdgOffset[0][stage_i], 0, true, true,
            false, true, B_row, K / (sizeof(BLdgType) / sizeof(T)), MACA_ICMP_SLT);
        __builtin_mxc_ldg_b128_bsm_predicator(
            WSM_Ldg + 0x4000 * stage_i + 0x3000, BPtr + BLdgOffset[1][stage_i], 0, true, true,
            false, true, B_row, K / (sizeof(BLdgType) / sizeof(T)), MACA_ICMP_SLT);
    }
    APtr += 128 * sizeof(T);
    BPtr += 128 * sizeof(T);
    K -= 128;

    arrive_gvmcnt(4 * (Stage - 1));
    __builtin_mxc_barrier_inst();

    // WSM_lds如果使用ALdsType*指针，会使得部分lds指令要重新计算 new_offset=offset*16
    uint8_t *WSM_lds = reinterpret_cast<uint8_t *>(&WSM[0]);

    a[0][0] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[0]);
    a[0][1] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[1]);
    a[0][2] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[2]);
    a[0][3] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[3]);
    b[0][0] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[0]);
    b[0][1] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[1]);
    b[0][2] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[2]);
    b[0][3] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[3]);

    arrive_gvmcnt(4 * (Stage - 2));
    __builtin_mxc_barrier_inst();

    a[1][0] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[0] + 0x4000);
    a[1][1] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[1] + 0x4000);
    a[1][2] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[2] + 0x4000);
    a[1][3] = *reinterpret_cast<ALdsType *>(WSM_lds + ALdsOffset[3] + 0x4000);
    b[1][0] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[0] + 0x4000);
    b[1][1] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[1] + 0x4000);
    b[1][2] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[2] + 0x4000);
    b[1][3] = *reinterpret_cast<ALdsType *>(WSM_lds + BLdsOffset[3] + 0x4000);

    // LDG not consider mask of K
    for (; K >= 128; K -= 128) {
        {
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][0][0], b[0][0][1], a[0][0][0], a[0][0][1], C_f32[0][0]);
            TILEOPS_MACA_A_LDG(WSM_Ldg + 0x4000 * 0 + 0x0000, APtr + ALdgOffset[0][0], false,
                               K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][0][2], b[0][0][3], a[0][0][2], a[0][0][3], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][1][0], b[0][1][1], a[0][1][0], a[0][1][1], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][1][2], b[0][1][3], a[0][1][2], a[0][1][3], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][2][0], b[0][2][1], a[0][2][0], a[0][2][1], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][2][2], b[0][2][3], a[0][2][2], a[0][2][3], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][3][0], b[0][3][1], a[0][3][0], a[0][3][1], C_f32[0][0]);
            C_f32[0][0] =
                mma_16x16x16b16<T>(b[0][3][2], b[0][3][3], a[0][3][2], a[0][3][3], C_f32[0][0]);

            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][0][0], b[0][0][1], a[1][0][0], a[1][0][1], C_f32[1][0]);
            TILEOPS_MACA_A_LDG(WSM_Ldg + 0x4000 * 0 + 0x1000, APtr + ALdgOffset[1][0], false,
                               K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][0][2], b[0][0][3], a[1][0][2], a[1][0][3], C_f32[1][0]);
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][1][0], b[0][1][1], a[1][1][0], a[1][1][1], C_f32[1][0]);
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][1][2], b[0][1][3], a[1][1][2], a[1][1][3], C_f32[1][0]);
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][2][0], b[0][2][1], a[1][2][0], a[1][2][1], C_f32[1][0]);
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][2][2], b[0][2][3], a[1][2][2], a[1][2][3], C_f32[1][0]);
            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][3][0], b[0][3][1], a[1][3][0], a[1][3][1], C_f32[1][0]);
            arrive_gvmcnt(4 * (Stage - 3) + 2);
            __builtin_mxc_barrier_inst();

            C_f32[1][0] =
                mma_16x16x16b16<T>(b[0][3][2], b[0][3][3], a[1][3][2], a[1][3][3], C_f32[1][0]);
            a[2][0] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + ALdsOffset[0]);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][0][0], b[1][0][1], a[0][0][0], a[0][0][1], C_f32[0][1]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 0 + 0x2000, BPtr + BLdgOffset[0][0])
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][0][2], b[1][0][3], a[0][0][2], a[0][0][3], C_f32[0][1]);
            a[2][1] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + ALdsOffset[1]);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][1][0], b[1][1][1], a[0][1][0], a[0][1][1], C_f32[0][1]);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][1][2], b[1][1][3], a[0][1][2], a[0][1][3], C_f32[0][1]);
            a[2][2] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + ALdsOffset[2]);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][2][0], b[1][2][1], a[0][2][0], a[0][2][1], C_f32[0][1]);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][2][2], b[1][2][3], a[0][2][2], a[0][2][3], C_f32[0][1]);
            a[2][3] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + ALdsOffset[3]);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][3][0], b[1][3][1], a[0][3][0], a[0][3][1], C_f32[0][1]);
            C_f32[0][1] =
                mma_16x16x16b16<T>(b[1][3][2], b[1][3][3], a[0][3][2], a[0][3][3], C_f32[0][1]);
            b[2][0] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + BLdsOffset[0]);

            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][0][0], b[1][0][1], a[1][0][0], a[1][0][1], C_f32[1][1]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 0 + 0x3000, BPtr + BLdgOffset[1][0])
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][0][2], b[1][0][3], a[1][0][2], a[1][0][3], C_f32[1][1]);
            b[2][1] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + BLdsOffset[1]);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][1][0], b[1][1][1], a[1][1][0], a[1][1][1], C_f32[1][1]);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][1][2], b[1][1][3], a[1][1][2], a[1][1][3], C_f32[1][1]);
            b[2][2] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + BLdsOffset[2]);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][2][0], b[1][2][1], a[1][2][0], a[1][2][1], C_f32[1][1]);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][2][2], b[1][2][3], a[1][2][2], a[1][2][3], C_f32[1][1]);
            b[2][3] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x8000 + BLdsOffset[3]);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][3][0], b[1][3][1], a[1][3][0], a[1][3][1], C_f32[1][1]);
            C_f32[1][1] =
                mma_16x16x16b16<T>(b[1][3][2], b[1][3][3], a[1][3][2], a[1][3][3], C_f32[1][1]);

            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][0][0], b[0][0][1], a[2][0][0], a[2][0][1], C_f32[2][0]);
            TILEOPS_MACA_A_LDG(WSM_Ldg + 0x4000 * 1 + 0x0000, APtr + ALdgOffset[0][1], false,
                               K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][0][2], b[0][0][3], a[2][0][2], a[2][0][3], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][1][0], b[0][1][1], a[2][1][0], a[2][1][1], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][1][2], b[0][1][3], a[2][1][2], a[2][1][3], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][2][0], b[0][2][1], a[2][2][0], a[2][2][1], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][2][2], b[0][2][3], a[2][2][2], a[2][2][3], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][3][0], b[0][3][1], a[2][3][0], a[2][3][1], C_f32[2][0]);
            C_f32[2][0] =
                mma_16x16x16b16<T>(b[0][3][2], b[0][3][3], a[2][3][2], a[2][3][3], C_f32[2][0]);

            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][0][0], b[1][0][1], a[2][0][0], a[2][0][1], C_f32[2][1]);
            TILEOPS_MACA_A_LDG(WSM_Ldg + 0x4000 * 1 + 0x1000, APtr + ALdgOffset[1][1], false,
                               K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][0][2], b[1][0][3], a[2][0][2], a[2][0][3], C_f32[2][1]);
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][1][0], b[1][1][1], a[2][1][0], a[2][1][1], C_f32[2][1]);
            arrive_gvmcnt(4 * (Stage - 4) + 6);
            __builtin_mxc_barrier_inst();
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][1][2], b[1][1][3], a[2][1][2], a[2][1][3], C_f32[2][1]);
            a[3][0] = *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + ALdsOffset[0]);
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][2][0], b[1][2][1], a[2][2][0], a[2][2][1], C_f32[2][1]);
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][2][2], b[1][2][3], a[2][2][2], a[2][2][3], C_f32[2][1]);
            a[3][1] = *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + ALdsOffset[1]);
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][3][0], b[1][3][1], a[2][3][0], a[2][3][1], C_f32[2][1]);
            C_f32[2][1] =
                mma_16x16x16b16<T>(b[1][3][2], b[1][3][3], a[2][3][2], a[2][3][3], C_f32[2][1]);
            a[3][2] = *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + ALdsOffset[2]);

            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][0][0], b[2][0][1], a[0][0][0], a[0][0][1], C_f32[0][2]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 1 + 0x2000, BPtr + BLdgOffset[0][1])
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][0][2], b[2][0][3], a[0][0][2], a[0][0][3], C_f32[0][2]);
            a[3][3] = *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + ALdsOffset[3]);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][1][0], b[2][1][1], a[0][1][0], a[0][1][1], C_f32[0][2]);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][1][2], b[2][1][3], a[0][1][2], a[0][1][3], C_f32[0][2]);
            b[3][0] = *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + BLdsOffset[0]);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][2][0], b[2][2][1], a[0][2][0], a[0][2][1], C_f32[0][2]);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][2][2], b[2][2][3], a[0][2][2], a[0][2][3], C_f32[0][2]);
            b[3][1] = *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + BLdsOffset[1]);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][3][0], b[2][3][1], a[0][3][0], a[0][3][1], C_f32[0][2]);
            C_f32[0][2] =
                mma_16x16x16b16<T>(b[2][3][2], b[2][3][3], a[0][3][2], a[0][3][3], C_f32[0][2]);
            b[3][2] = *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + BLdsOffset[2]);

            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][0][0], b[2][0][1], a[1][0][0], a[1][0][1], C_f32[1][2]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 1 + 0x3000, BPtr + BLdgOffset[1][1])
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][0][2], b[2][0][3], a[1][0][2], a[1][0][3], C_f32[1][2]);
            b[3][3] = *reinterpret_cast<ALdsType *>(WSM_lds + 0xC000 + BLdsOffset[3]);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][1][0], b[2][1][1], a[1][1][0], a[1][1][1], C_f32[1][2]);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][1][2], b[2][1][3], a[1][1][2], a[1][1][3], C_f32[1][2]);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][2][0], b[2][2][1], a[1][2][0], a[1][2][1], C_f32[1][2]);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][2][2], b[2][2][3], a[1][2][2], a[1][2][3], C_f32[1][2]);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][3][0], b[2][3][1], a[1][3][0], a[1][3][1], C_f32[1][2]);
            C_f32[1][2] =
                mma_16x16x16b16<T>(b[2][3][2], b[2][3][3], a[1][3][2], a[1][3][3], C_f32[1][2]);

            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][0][0], b[2][0][1], a[2][0][0], a[2][0][1], C_f32[2][2]);
            TILEOPS_MACA_A_LDG(WSM_Ldg + 0x4000 * 2 + 0x0000, APtr + ALdgOffset[0][2], false,
                               K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][0][2], b[2][0][3], a[2][0][2], a[2][0][3], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][1][0], b[2][1][1], a[2][1][0], a[2][1][1], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][1][2], b[2][1][3], a[2][1][2], a[2][1][3], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][2][0], b[2][2][1], a[2][2][0], a[2][2][1], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][2][2], b[2][2][3], a[2][2][2], a[2][2][3], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][3][0], b[2][3][1], a[2][3][0], a[2][3][1], C_f32[2][2]);
            C_f32[2][2] =
                mma_16x16x16b16<T>(b[2][3][2], b[2][3][3], a[2][3][2], a[2][3][3], C_f32[2][2]);

            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][0][0], b[0][0][1], a[3][0][0], a[3][0][1], C_f32[3][0]);
            TILEOPS_MACA_A_LDG(WSM_Ldg + 0x4000 * 2 + 0x1000, APtr + ALdgOffset[1][2], false,
                               K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][0][2], b[0][0][3], a[3][0][2], a[3][0][3], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][1][0], b[0][1][1], a[3][1][0], a[3][1][1], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][1][2], b[0][1][3], a[3][1][2], a[3][1][3], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][2][0], b[0][2][1], a[3][2][0], a[3][2][1], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][2][2], b[0][2][3], a[3][2][2], a[3][2][3], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][3][0], b[0][3][1], a[3][3][0], a[3][3][1], C_f32[3][0]);
            C_f32[3][0] =
                mma_16x16x16b16<T>(b[0][3][2], b[0][3][3], a[3][3][2], a[3][3][3], C_f32[3][0]);
            arrive_gvmcnt(4 * (Stage - 5) + 10);
            __builtin_mxc_barrier_inst();

            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][0][0], b[3][0][1], a[0][0][0], a[0][0][1], C_f32[0][3]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 2 + 0x2000, BPtr + BLdgOffset[0][2])
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][0][2], b[3][0][3], a[0][0][2], a[0][0][3], C_f32[0][3]);
            b[0][0] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[0]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][1][0], b[3][1][1], a[0][1][0], a[0][1][1], C_f32[0][3]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][1][2], b[3][1][3], a[0][1][2], a[0][1][3], C_f32[0][3]);
            b[0][1] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[1]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][2][0], b[3][2][1], a[0][2][0], a[0][2][1], C_f32[0][3]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][2][2], b[3][2][3], a[0][2][2], a[0][2][3], C_f32[0][3]);
            b[0][2] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[2]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][3][0], b[3][3][1], a[0][3][0], a[0][3][1], C_f32[0][3]);
            C_f32[0][3] =
                mma_16x16x16b16<T>(b[3][3][2], b[3][3][3], a[0][3][2], a[0][3][3], C_f32[0][3]);
            b[0][3] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + BLdsOffset[3]);

            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][0][0], b[1][0][1], a[3][0][0], a[3][0][1], C_f32[3][1]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 2 + 0x3000, BPtr + BLdgOffset[1][2])
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][0][2], b[1][0][3], a[3][0][2], a[3][0][3], C_f32[3][1]);
            a[0][0] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[0]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][1][0], b[1][1][1], a[3][1][0], a[3][1][1], C_f32[3][1]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][1][2], b[1][1][3], a[3][1][2], a[3][1][3], C_f32[3][1]);
            a[0][1] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[1]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][2][0], b[1][2][1], a[3][2][0], a[3][2][1], C_f32[3][1]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][2][2], b[1][2][3], a[3][2][2], a[3][2][3], C_f32[3][1]);
            a[0][2] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[2]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][3][0], b[1][3][1], a[3][3][0], a[3][3][1], C_f32[3][1]);
            C_f32[3][1] =
                mma_16x16x16b16<T>(b[1][3][2], b[1][3][3], a[3][3][2], a[3][3][3], C_f32[3][1]);
            a[0][3] = *reinterpret_cast<ALdsType *>(WSM_lds + 0 + ALdsOffset[3]);

            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][0][0], b[3][0][1], a[1][0][0], a[1][0][1], C_f32[1][3]);
            TILEOPS_MACA_A_LDG(WSM_Ldg + 0x4000 * 3 + 0x0000, APtr + ALdgOffset[0][3], false,
                               K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][0][2], b[3][0][3], a[1][0][2], a[1][0][3], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][1][0], b[3][1][1], a[1][1][0], a[1][1][1], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][1][2], b[3][1][3], a[1][1][2], a[1][1][3], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][2][0], b[3][2][1], a[1][2][0], a[1][2][1], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][2][2], b[3][2][3], a[1][2][2], a[1][2][3], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][3][0], b[3][3][1], a[1][3][0], a[1][3][1], C_f32[1][3]);
            C_f32[1][3] =
                mma_16x16x16b16<T>(b[3][3][2], b[3][3][3], a[1][3][2], a[1][3][3], C_f32[1][3]);

            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][0][0], b[2][0][1], a[3][0][0], a[3][0][1], C_f32[3][2]);
            TILEOPS_MACA_A_LDG(WSM_Ldg + 0x4000 * 3 + 0x1000, APtr + ALdgOffset[1][3], false,
                               K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][0][2], b[2][0][3], a[3][0][2], a[3][0][3], C_f32[3][2]);
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][1][0], b[2][1][1], a[3][1][0], a[3][1][1], C_f32[3][2]);
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][1][2], b[2][1][3], a[3][1][2], a[3][1][3], C_f32[3][2]);
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][2][0], b[2][2][1], a[3][2][0], a[3][2][1], C_f32[3][2]);
            arrive_gvmcnt(4 * (Stage - 6) + 14);
            __builtin_mxc_barrier_inst();
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][2][2], b[2][2][3], a[3][2][2], a[3][2][3], C_f32[3][2]);
            a[1][0] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + ALdsOffset[0]);
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][3][0], b[2][3][1], a[3][3][0], a[3][3][1], C_f32[3][2]);
            C_f32[3][2] =
                mma_16x16x16b16<T>(b[2][3][2], b[2][3][3], a[3][3][2], a[3][3][3], C_f32[3][2]);
            a[1][1] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + ALdsOffset[1]);

            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][0][0], b[3][0][1], a[2][0][0], a[2][0][1], C_f32[2][3]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 3 + 0x2000, BPtr + BLdgOffset[0][3])
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][0][2], b[3][0][3], a[2][0][2], a[2][0][3], C_f32[2][3]);
            a[1][2] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + ALdsOffset[2]);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][1][0], b[3][1][1], a[2][1][0], a[2][1][1], C_f32[2][3]);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][1][2], b[3][1][3], a[2][1][2], a[2][1][3], C_f32[2][3]);
            a[1][3] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + ALdsOffset[3]);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][2][0], b[3][2][1], a[2][2][0], a[2][2][1], C_f32[2][3]);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][2][2], b[3][2][3], a[2][2][2], a[2][2][3], C_f32[2][3]);
            b[1][0] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + BLdsOffset[0]);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][3][0], b[3][3][1], a[2][3][0], a[2][3][1], C_f32[2][3]);
            C_f32[2][3] =
                mma_16x16x16b16<T>(b[3][3][2], b[3][3][3], a[2][3][2], a[2][3][3], C_f32[2][3]);
            b[1][1] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + BLdsOffset[1]);

            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][0][0], b[3][0][1], a[3][0][0], a[3][0][1], C_f32[3][3]);
            LDG_B128_BSM_NO_PREDICATOR(WSM_Ldg + 0x4000 * 3 + 0x3000, BPtr + BLdgOffset[1][3])
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][0][2], b[3][0][3], a[3][0][2], a[3][0][3], C_f32[3][3]);
            b[1][2] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + BLdsOffset[2]);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][1][0], b[3][1][1], a[3][1][0], a[3][1][1], C_f32[3][3]);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][1][2], b[3][1][3], a[3][1][2], a[3][1][3], C_f32[3][3]);
            b[1][3] = *reinterpret_cast<ALdsType *>(WSM_lds + 0x4000 + BLdsOffset[3]);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][2][0], b[3][2][1], a[3][2][0], a[3][2][1], C_f32[3][3]);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][2][2], b[3][2][3], a[3][2][2], a[3][2][3], C_f32[3][3]);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][3][0], b[3][3][1], a[3][3][0], a[3][3][1], C_f32[3][3]);
            C_f32[3][3] =
                mma_16x16x16b16<T>(b[3][3][2], b[3][3][3], a[3][3][2], a[3][3][3], C_f32[3][3]);
        }
        APtr += 128 * sizeof(T);
        BPtr += 128 * sizeof(T) * ldb / ldb;
        // add 256 will cause using private memory in this kernel, optmize in the future
        // BPtr += 256;
    }

    // LDG consider mask of K
    // 这里其实有lds冗余，但是不展开不太好调整lds指令，要确保a[0]和b[0]用完才能lds(a[0]/b[0])
    if (K > 0) {
#pragma unroll
        for (int stage_i = 0; stage_i < Stage; ++stage_i) {
            const int ldsIdx = (stage_i + 1) % Stage;
            uint8_t *WSM_lds2 = WSM_lds + (0x4000 * ldsIdx);

            for (int i = 0; i < stage_i; ++i) {
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][0][0], b[i][0][1], a[stage_i][0][0],
                                                           a[stage_i][0][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][0][2], b[i][0][3], a[stage_i][0][2],
                                                           a[stage_i][0][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][1][0], b[i][1][1], a[stage_i][1][0],
                                                           a[stage_i][1][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][1][2], b[i][1][3], a[stage_i][1][2],
                                                           a[stage_i][1][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][2][0], b[i][2][1], a[stage_i][2][0],
                                                           a[stage_i][2][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][2][2], b[i][2][3], a[stage_i][2][2],
                                                           a[stage_i][2][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][3][0], b[i][3][1], a[stage_i][3][0],
                                                           a[stage_i][3][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][3][2], b[i][3][3], a[stage_i][3][2],
                                                           a[stage_i][3][3], C_f32[stage_i][i + 0]);
            }
            for (int i = 0; i <= stage_i; ++i) {
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][0][0], b[stage_i][0][1], a[i][0][0], a[i][0][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][0][2], b[stage_i][0][3], a[i][0][2], a[i][0][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][1][0], b[stage_i][1][1], a[i][1][0], a[i][1][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][1][2], b[stage_i][1][3], a[i][1][2], a[i][1][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][2][0], b[stage_i][2][1], a[i][2][0], a[i][2][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][2][2], b[stage_i][2][3], a[i][2][2], a[i][2][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][3][0], b[stage_i][3][1], a[i][3][0], a[i][3][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][3][2], b[stage_i][3][3], a[i][3][2], a[i][3][3],
                                       C_f32[i][stage_i + 0]);
            }

            arrive_gvmcnt(4 * (Stage - 2));
            __builtin_mxc_barrier_inst();

            TILEOPS_MACA_A_LDG(
                WSM_Ldg + 0x4000 * stage_i + 0x0000, APtr + ALdgOffset[0][stage_i], true,
                K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            TILEOPS_MACA_A_LDG(
                WSM_Ldg + 0x4000 * stage_i + 0x1000, APtr + ALdgOffset[1][stage_i], true,
                K / (sizeof(ALdgType) / sizeof(T)), MACA_ICMP_SLT);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * stage_i + 0x2000, BPtr + BLdgOffset[0][stage_i], 0, true, true,
                false, true, B_row, K / (sizeof(BLdgType) / sizeof(T)), MACA_ICMP_SLT);
            __builtin_mxc_ldg_b128_bsm_predicator(
                WSM_Ldg + 0x4000 * stage_i + 0x3000, BPtr + BLdgOffset[1][stage_i], 0, true, true,
                false, true, B_row, K / (sizeof(BLdgType) / sizeof(T)), MACA_ICMP_SLT);

            a[ldsIdx][0] = *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[0]);
            a[ldsIdx][1] = *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[1]);
            a[ldsIdx][2] = *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[2]);
            a[ldsIdx][3] = *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[3]);
            b[ldsIdx][0] = *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[0]);
            b[ldsIdx][1] = *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[1]);
            b[ldsIdx][2] = *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[2]);
            b[ldsIdx][3] = *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[3]);
        }
    }

    // only do LDA and MMA, no LDGs
    {
#pragma unroll
        for (int stage_i = 0; stage_i < Stage; ++stage_i) {
            const int ldsIdx = (stage_i + 1) % Stage;
            uint8_t *WSM_lds2 = WSM_lds + (0x4000 * ldsIdx);

            for (int i = 0; i < stage_i; ++i) {
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][0][0], b[i][0][1], a[stage_i][0][0],
                                                           a[stage_i][0][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][0][2], b[i][0][3], a[stage_i][0][2],
                                                           a[stage_i][0][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][1][0], b[i][1][1], a[stage_i][1][0],
                                                           a[stage_i][1][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][1][2], b[i][1][3], a[stage_i][1][2],
                                                           a[stage_i][1][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][2][0], b[i][2][1], a[stage_i][2][0],
                                                           a[stage_i][2][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][2][2], b[i][2][3], a[stage_i][2][2],
                                                           a[stage_i][2][3], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][3][0], b[i][3][1], a[stage_i][3][0],
                                                           a[stage_i][3][1], C_f32[stage_i][i + 0]);
                C_f32[stage_i][i + 0] = mma_16x16x16b16<T>(b[i][3][2], b[i][3][3], a[stage_i][3][2],
                                                           a[stage_i][3][3], C_f32[stage_i][i + 0]);
            }
            for (int i = 0; i <= stage_i; ++i) {
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][0][0], b[stage_i][0][1], a[i][0][0], a[i][0][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][0][2], b[stage_i][0][3], a[i][0][2], a[i][0][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][1][0], b[stage_i][1][1], a[i][1][0], a[i][1][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][1][2], b[stage_i][1][3], a[i][1][2], a[i][1][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][2][0], b[stage_i][2][1], a[i][2][0], a[i][2][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][2][2], b[stage_i][2][3], a[i][2][2], a[i][2][3],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][3][0], b[stage_i][3][1], a[i][3][0], a[i][3][1],
                                       C_f32[i][stage_i + 0]);
                C_f32[i][stage_i + 0] =
                    mma_16x16x16b16<T>(b[stage_i][3][2], b[stage_i][3][3], a[i][3][2], a[i][3][3],
                                       C_f32[i][stage_i + 0]);
            }

            if (stage_i == 0) {
                arrive_gvmcnt(4 * (Stage - 2 - 0));
                __builtin_mxc_barrier_inst();
            } else if (stage_i == 1) {
                arrive_gvmcnt(4 * (Stage - 2 - 1));
                __builtin_mxc_barrier_inst();
            } else if (stage_i == 2) {
                arrive_gvmcnt(4 * (Stage - 2 - 2));
                __builtin_mxc_barrier_inst();
            }

            if (stage_i < Stage - 1) {
                a[ldsIdx][0] = *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[0]);
                a[ldsIdx][1] = *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[1]);
                a[ldsIdx][2] = *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[2]);
                a[ldsIdx][3] = *reinterpret_cast<ALdsType *>(WSM_lds2 + ALdsOffset[3]);
                b[ldsIdx][0] = *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[0]);
                b[ldsIdx][1] = *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[1]);
                b[ldsIdx][2] = *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[2]);
                b[ldsIdx][3] = *reinterpret_cast<ALdsType *>(WSM_lds2 + BLdsOffset[3]);
            }
        }
    }

    /*
     * layout of C:
     *       col0       col1       col2       col3       col4
     * row0 C[0][0]+0  C[0][1]+0  C[0][2]+0  C[0][3]+0  C[0][0]+1
     * row1 C[1][0]+0
     * row2 C[2][0]+0
     * row3 C[3][0]+0
     */

    const int C_row = (tid & 15) + (slot / 2) * 16;
    const int C_col = (lane / 16 * 16) + (slot & 1) * 64;
    const int CStgOffset = C_row + C_col * ldc;
    const bool C_row_flag = C_row < M_C;

    for (int i = 0; i < 4; ++i) {
        for (int j2 = 0; j2 < 4; ++j2) {
            const bool C_col_flag = i * 4 + j2 < N_C;
            const bool stgFlag = C_row_flag && C_col_flag;

            if (stgFlag) {
                float C_f32_res[4];
                for (int j = 0; j < 4; ++j) {
                    C_f32_res[j] = C_f32[j][j2][i] * static_cast<float>(alpha);
                }

                if constexpr (!IsBetaZero) {
                    CStgType C_tmp5 = CPtr[CStgOffset];
                    Tc *C_tmp6 = reinterpret_cast<Tc *>(&C_tmp5);
                    for (int j = 0; j < 4; ++j) {
                        C_f32_res[j] += static_cast<float>(C_tmp6[j]) * static_cast<float>(beta);
                    }
                }

                if constexpr (std::is_same<Tc, __half>::value ||
                              std::is_same<Tc, maca_bfloat16>::value) {
                    /*
                     *  inplace cvt will cost private memory, following is not good
                     *  Tc *C_half_tmp = reinterpret_cast<Tc*>(&C_tmp[0]);
                     */
                    Tc C_half_tmp[4] = {0};
                    for (int j = 0; j < 4; ++j) {
                        C_half_tmp[j] = static_cast<Tc>(C_f32_res[j]);
                    }

                    if constexpr (UseAtomicAdd) {
                        static_assert(std::is_same<Tc, __half>::value,
                                      "split-k atomic writeback is only implemented for fp16");
                        __half2 *C_half2 = reinterpret_cast<__half2 *>(CPtr);
                        atomicAdd(C_half2 + CStgOffset * 2 + 0,
                                  __halves2half2(C_half_tmp[0], C_half_tmp[1]));
                        atomicAdd(C_half2 + CStgOffset * 2 + 1,
                                  __halves2half2(C_half_tmp[2], C_half_tmp[3]));
                    } else {
                        uint *C_half2_tmp = reinterpret_cast<uint *>(&C_f32_res[0]);
                        uint16_t *C_halfToB16_tmp = reinterpret_cast<uint16_t *>(&C_half_tmp[0]);
                        C_half2_tmp[0] = static_cast<uint>(C_halfToB16_tmp[0]) +
                                         (static_cast<uint>(C_halfToB16_tmp[1]) << 16);
                        C_half2_tmp[1] = static_cast<uint>(C_halfToB16_tmp[2]) +
                                         (static_cast<uint>(C_halfToB16_tmp[3]) << 16);
                        CPtr[CStgOffset] = *reinterpret_cast<CStgType *>(&C_f32_res[0]);
                    }
                } else {
                    if constexpr (UseAtomicAdd && std::is_same<Tc, float>::value) {
                        float *C_float = reinterpret_cast<float *>(CPtr);
                        atomicAdd(C_float + CStgOffset * 4 + 0, C_f32_res[0]);
                        atomicAdd(C_float + CStgOffset * 4 + 1, C_f32_res[1]);
                        atomicAdd(C_float + CStgOffset * 4 + 2, C_f32_res[2]);
                        atomicAdd(C_float + CStgOffset * 4 + 3, C_f32_res[3]);
                    } else {
                        if constexpr (UseAtomicAdd) {
                            static_assert(!UseAtomicAdd,
                                          "split-k atomic writeback requires fp16 or fp32 output");
                        }
                        CPtr[CStgOffset] = *reinterpret_cast<CStgType *>(&C_f32_res[0]);
                    }
                }
            }
            CPtr += ldc;
        }
    }
}

template <typename T, typename Tc, typename Tscal, bool IsBetaZero>
__global__ void hgemm_tn_128x128x128_4m1n8k_256t(const void *A,
                                                 const void *B,
                                                 void *C,
                                                 int M,
                                                 int N,
                                                 int K,
                                                 int lda,
                                                 int ldb,
                                                 int ldc,
                                                 Tscal alpha,
                                                 Tscal beta) {
        hgemm_tn_128x128x128_4m1n8k_256t_device<T, Tc, Tscal, IsBetaZero>(
            A, B, C, M, N, K, lda, ldb, ldc, alpha, beta, blockIdx.x, blockIdx.y);
}

template <typename T, typename Tc, typename Tscal, bool IsBetaZero>
__global__ void hgemm_tn_128x128x128_4m1n8k_256t_atomic(const void *A,
                                                         const void *B,
                                                         void *C,
                                                         int M,
                                                         int N,
                                                         int K,
                                                         int lda,
                                                         int ldb,
                                                         int ldc,
                                                         Tscal alpha,
                                                         Tscal beta) {
    hgemm_tn_128x128x128_4m1n8k_256t_device<T, Tc, Tscal, IsBetaZero, true>(
        A, B, C, M, N, K, lda, ldb, ldc, alpha, beta, blockIdx.x, blockIdx.y);
}
