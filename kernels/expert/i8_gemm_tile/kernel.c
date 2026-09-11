/*
 * i8_gemm_tile HVX expert kernel v1
 *
 * C[i*N+j] = sum_k A[i*K+k]*B[k*N+j], int32 accumulate.
 * A is [M x K], B is [K x N], C is [M x N] int32. Shapes: M=32, N=32, K=130.
 * Row-major; B is [K,N] (NOT pre-transposed -- see nearmiss_transB).
 *
 * Strategy (proven on i8_gemm_bias, 3.35x): vectorize across the N=32 output
 * columns. One HVX vector (128B = 32 x int32) holds one output row. Reduce K in
 * groups of 4 via signed vrmpy Q6_Vw_vrmpyacc_VwVbVb:
 *   acc[j] += A[i][k..k+3] . B[k..k+3][j]
 * Pre-pack B into BT_pack[k4][j*4+r] = B[k4*4+r][j] so each k-group is one 128B
 * vector with byte 4*j+r = B[k4*4+r][j]. K padded to a multiple of 4 (zero pad).
 */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C,
                      int M, int N, int K) {
    int Kpad = (K + 3) & ~3;
    int ngroups = Kpad / 4;

    /* BT_pack[k4 * (N*4) + j*4 + r] = B[(k4*4+r)*N + j]; N*4=128B per k-group. */
    static int8_t BT_pack[((256 + 3) / 4) * 128]; /* room for Kpad<=256, N<=32 */
    memset(BT_pack, 0, (size_t)ngroups * N * 4);
    for (int k = 0; k < K; k++) {
        int k4 = k / 4, r = k % 4;
        const int8_t *Brow = B + k * N;
        int8_t *dst = BT_pack + k4 * N * 4 + r;
        for (int j = 0; j < N; j++)
            dst[j * 4] = Brow[j];
    }

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;
        int32_t *Crow = C + i * N;
        HVX_Vector vacc = Q6_V_vzero();

        for (int k4 = 0; k4 < ngroups; k4++) {
            int kb = k4 * 4;
            uint32_t a0 = (kb + 0 < K) ? (uint8_t)Arow[kb + 0] : 0;
            uint32_t a1 = (kb + 1 < K) ? (uint8_t)Arow[kb + 1] : 0;
            uint32_t a2 = (kb + 2 < K) ? (uint8_t)Arow[kb + 2] : 0;
            uint32_t a3 = (kb + 3 < K) ? (uint8_t)Arow[kb + 3] : 0;
            /* Word w = [a0,a1,a2,a3] (byte0=a0); splat to all 32 word lanes so
               lane j holds A[i][k..k+3] -- matches vb byte 4*j+r = B[k+r][j]. */
            uint32_t w = a0 | (a1 << 8) | (a2 << 16) | (a3 << 24);
            HVX_Vector va = Q6_V_vsplat_R(w);
            HVX_Vector vb = *(const HVX_UVector *)(BT_pack + k4 * N * 4);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, va, vb);
        }
        *(HVX_Vector *)Crow = vacc;
    }
}
