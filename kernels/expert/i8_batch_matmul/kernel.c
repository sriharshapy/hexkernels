/*
 * i8_batch_matmul HVX expert kernel v1
 * BATCH=4 independent GEMMs: Cb[i*N+j] = sum_k Ab[i*K+k]*Bb[k*N+j], int32 acc.
 * Per batch M=16, N=16, K=130. Reuses the vrmpy-tile idiom (BT_pack + vsplat +
 * Q6_Vw_vrmpyacc_VwVbVb) proven on i8_gemm_tile (10x). N=16 -> each output row is
 * 16 int32 = 64 bytes, so we store only the low 64 bytes of the accumulator.
 */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#define BATCH 4

void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C,
                      int M, int N, int K) {
    int Kpad = (K + 3) & ~3;
    int ngroups = Kpad / 4;
    static int8_t BT_pack[((256 + 3) / 4) * 128];
    int32_t rowbuf[32] __attribute__((aligned(128)));

    for (int b = 0; b < BATCH; b++) {
        const int8_t *Ab = A + b * M * K;
        const int8_t *Bb = B + b * K * N;
        int32_t      *Cb = C + b * M * N;

        /* pack Bb[K,N] -> BT_pack: byte (k4*N*4 + j*4 + r) = Bb[(k4*4+r)*N + j] */
        memset(BT_pack, 0, (size_t)ngroups * N * 4);
        for (int k = 0; k < K; k++) {
            int k4 = k / 4, r = k % 4;
            const int8_t *Brow = Bb + k * N;
            int8_t *dst = BT_pack + k4 * N * 4 + r;
            for (int j = 0; j < N; j++)
                dst[j * 4] = Brow[j];
        }

        for (int i = 0; i < M; i++) {
            const int8_t *Arow = Ab + i * K;
            HVX_Vector vacc = Q6_V_vzero();
            for (int k4 = 0; k4 < ngroups; k4++) {
                int kb = k4 * 4;
                uint32_t a0 = (kb+0 < K) ? (uint8_t)Arow[kb+0] : 0;
                uint32_t a1 = (kb+1 < K) ? (uint8_t)Arow[kb+1] : 0;
                uint32_t a2 = (kb+2 < K) ? (uint8_t)Arow[kb+2] : 0;
                uint32_t a3 = (kb+3 < K) ? (uint8_t)Arow[kb+3] : 0;
                HVX_Vector va = Q6_V_vsplat_R(a0 | (a1<<8) | (a2<<16) | (a3<<24));
                HVX_Vector vb = *(const HVX_UVector *)(BT_pack + k4 * N * 4);
                vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, va, vb);
            }
            *(HVX_Vector *)rowbuf = vacc;         /* 32 int32 in aligned buf */
            for (int j = 0; j < N; j++) Cb[i*N+j] = rowbuf[j];  /* store N of them */
        }
    }
}
