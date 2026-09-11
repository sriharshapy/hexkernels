/*
 * gemm_tile_i8 Solution 1: tile-vectorized k4-group vrmpy accumulation.
 *
 * C[i*N+j] = sum_k A[i*K+k]*B[j*K+k]. A is [M x K], B is [N x K] (TRANSPOSED
 * layout -- row j of B is contiguous over k, just like row i of A). Shapes:
 * M=24, N=24, K=100.
 *
 * Strategy (adapted from i8_gemm_tile's proven vrmpy tile trick, re-derived for
 * the transposed-B layout): vectorize across the N=24 output columns of one row
 * at a time. One HVX vector (128B = 32 x int32, first N=24 lanes meaningful)
 * holds one output row's accumulator. Reduce K in groups of 4 via signed vrmpy
 * Q6_Vw_vrmpyacc_VwVbVb: acc[j] += A[i][k..k+3] . B[j][k..k+3].
 *
 * Packing (transpose step, done once): BT_pack[k4*128 + j*4+r] = B[j*K + k4*4+r]
 * for j in [0,N); each 128B row covers one k-group across all (padded-to-32) j.
 * K padded to a multiple of 4 (zero pad); N padded to 32 in the pack (zero pad;
 * the extra lanes are never read back out to C, so their content doesn't matter,
 * zero-fill just keeps them deterministic).
 */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C,
                      int M, int N, int K) {
    int Kpad = (K + 3) & ~3;
    int ngroups = Kpad / 4;

    /* BT_pack[k4*128 + j*4+r] = B[j*K + (k4*4+r)]; 128B per k-group (32 lanes,
     * only the first N are real data, the rest zero). Room for Kpad<=128 (32
     * groups), N<=32. */
    static int8_t __attribute__((aligned(128))) BT_pack[32 * 128];
    memset(BT_pack, 0, (size_t)ngroups * 128);
    for (int k = 0; k < K; k++) {
        int k4 = k / 4, r = k % 4;
        int8_t *dst = BT_pack + k4 * 128 + r;
        for (int j = 0; j < N; j++)
            dst[j * 4] = B[j * K + k];
    }

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;
        HVX_Vector vacc = Q6_V_vzero();

        for (int k4 = 0; k4 < ngroups; k4++) {
            int kb = k4 * 4;
            uint32_t a0 = (kb + 0 < K) ? (uint8_t)Arow[kb + 0] : 0;
            uint32_t a1 = (kb + 1 < K) ? (uint8_t)Arow[kb + 1] : 0;
            uint32_t a2 = (kb + 2 < K) ? (uint8_t)Arow[kb + 2] : 0;
            uint32_t a3 = (kb + 3 < K) ? (uint8_t)Arow[kb + 3] : 0;
            uint32_t w = a0 | (a1 << 8) | (a2 << 16) | (a3 << 24);
            HVX_Vector va = Q6_V_vsplat_R(w);
            HVX_Vector vb = *(const HVX_Vector *)(BT_pack + k4 * 128);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, va, vb);
        }

        /* Only the first N=24 of 32 lanes are real output -- store via a temp
         * buffer + narrow copy so we never overwrite the next row's cells
         * (C's row stride is N*4=96B, not the full 128B vector width). */
        int32_t tmp[32] __attribute__((aligned(128)));
        *(HVX_Vector *)tmp = vacc;
        memcpy(C + i * N, tmp, (size_t)N * sizeof(int32_t));
    }
}
