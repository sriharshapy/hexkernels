/*
 * HVX batched int8 matmul + bias -> int32.
 * BATCH=4, M=16, N=16, K=32.
 *
 * Strategy:
 *   Pre-transpose B per batch into Bt so word lane j holds
 *   [B[k0][j], B[k0+1][j], B[k0+2][j], B[k0+3][j]] for k-groups of 4.
 *   N=16 => 16 word lanes = 64 bytes; K=32 => 8 k-groups per batch.
 *   For each (batch, row i):
 *     splat A[i][k0..k3] into int32 across all lanes;
 *     vrmpy accumulates all 16 columns at once.
 *   Add bias (scalar copy into aligned buf once per batch).
 *   Write 16 int32 outputs via predicated 64-byte vmem store.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

#define BATCH 4

void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, int32_t *C,
                      int M, int N, int K)
{
    /* Stack-allocated transposed B: 4 batches × 8 kgroups × 128 bytes = 4096 */
    int8_t __attribute__((aligned(128))) Bt[BATCH * 8 * 128];

    const int num_kg = K >> 2;   /* K=32 => 8 */

    /* Build transposed B for all batches */
    for (int b = 0; b < BATCH; b++) {
        const int8_t *Bb    = B + b * K * N;
        int8_t       *Bt_b  = Bt + b * (8 * 128);
        for (int kg = 0; kg < num_kg; kg++) {
            int8_t * restrict blk = Bt_b + kg * 128;
            int k0 = kg * 4;
            /* Zero entire 128-byte block; only lanes 0..15 used (j=0..15) */
            memset(blk, 0, 128);
            for (int j = 0; j < N; j++) {
                blk[j*4+0] = Bb[(k0+0)*N+j];
                blk[j*4+1] = Bb[(k0+1)*N+j];
                blk[j*4+2] = Bb[(k0+2)*N+j];
                blk[j*4+3] = Bb[(k0+3)*N+j];
            }
        }
    }

    for (int b = 0; b < BATCH; b++) {
        const int8_t  *Ab   = A    + b * M * K;
        const int32_t *bib  = bias + b * N;
        int32_t       *Cb   = C    + b * M * N;
        const int8_t  *Bt_b = Bt   + b * (8 * 128);

        /* Load bias into aligned 32-word buffer (bib is not always 128B aligned for b>0) */
        int32_t biasbuf[32] __attribute__((aligned(128)));
        memset(biasbuf, 0, 128);
        for (int j = 0; j < N; j++) biasbuf[j] = bib[j];
        HVX_Vector vbias = *(const HVX_Vector *)biasbuf;

        for (int i = 0; i < M; i++) {
            const int8_t *Arow = Ab + i * K;

            HVX_Vector acc = Q6_V_vzero();

            for (int kg = 0; kg < num_kg; kg++) {
                int k0 = kg * 4;
                int32_t a4 = (int32_t)((uint8_t)Arow[k0+0])
                           | (int32_t)((uint8_t)Arow[k0+1] << 8)
                           | (int32_t)((uint8_t)Arow[k0+2] << 16)
                           | (int32_t)((uint8_t)Arow[k0+3] << 24);
                HVX_Vector va = Q6_V_vsplat_R(a4);
                HVX_Vector vb = *(const HVX_Vector *)(Bt_b + kg * 128);
                acc = Q6_Vw_vrmpyacc_VwVbVb(acc, va, vb);
            }

            HVX_Vector vout = Q6_Vw_vadd_VwVw(acc, vbias);

            /* Extract results to aligned tmp buffer, then scalar-copy to Cb */
            int32_t tmp[32] __attribute__((aligned(128)));
            *(HVX_Vector *)tmp = vout;
            int32_t *row = Cb + i * N;
            for (int j = 0; j < N; j++)
                row[j] = tmp[j];
        }
    }
}
