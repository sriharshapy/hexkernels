/*
 * HVX fused GEMM-softmax tile: GM=GN=GK=16.
 *
 * Step A: A[16x16] * B[16x16] -> int32 acc (vrmpy 4-wide)
 * Step B: saturating clamp to int8 scores
 * Step C: rowwise integer softmax using exp_lut -> uint8 out
 *
 * GEMM strategy:
 *   Pretranspose B into Bt[4 k-groups][128B each].
 *   Each 128B block has 32 word lanes; lanes 0..15 hold B[j][k0..k0+3] for j=0..15.
 *   For each row i: splat A[i][k0..k0+3] as int32, vrmpyacc into 128B acc vector.
 *   Extract 16 int32 accumulators, clamp to int8 scores.
 *
 * Softmax: scalar (GN=16 is tiny, GEMM is the cycle cost).
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

#define GM 16
#define GN 16
#define GK 16

void candidate_kernel(const int8_t *A, const int8_t *B,
                      uint8_t *out,
                      const uint8_t *exp_lut)
{
    /* Transposed B: 4 k-groups x 128 bytes.
     * Bt[kg][j*4+q] = B[(kg*4+q)*GN + j] for j=0..GN-1, q=0..3.
     * Upper 16 word lanes (j=16..31) are zero-padded. */
    int8_t Bt[4 * 128] __attribute__((aligned(128)));

    /* Build Bt */
    for (int kg = 0; kg < 4; kg++) {
        int8_t *blk = Bt + kg * 128;
        memset(blk, 0, 128);
        int k0 = kg * 4;
        for (int j = 0; j < GN; j++) {
            blk[j*4+0] = B[(k0+0)*GN + j];
            blk[j*4+1] = B[(k0+1)*GN + j];
            blk[j*4+2] = B[(k0+2)*GN + j];
            blk[j*4+3] = B[(k0+3)*GN + j];
        }
    }

    /* Accumulator buffer: 32 int32 lanes (128B), we use lanes 0..15 */
    int32_t accbuf[32] __attribute__((aligned(128)));

    /* Scores tile: GM x GN = 256 bytes */
    int8_t scores[GM * GN] __attribute__((aligned(128)));

    /* Step A+B: GEMM -> int8 scores */
    for (int i = 0; i < GM; i++) {
        const int8_t *Arow = A + i * GK;

        HVX_Vector acc = Q6_V_vzero();

        /* 4 k-groups of 4 bytes each */
        for (int kg = 0; kg < 4; kg++) {
            int k0 = kg * 4;
            int32_t a4 = (int32_t)((uint8_t)Arow[k0+0])
                       | (int32_t)((uint8_t)Arow[k0+1] << 8)
                       | (int32_t)((uint8_t)Arow[k0+2] << 16)
                       | (int32_t)((uint8_t)Arow[k0+3] << 24);
            HVX_Vector va = Q6_V_vsplat_R(a4);
            HVX_Vector vb = *(const HVX_Vector *)(Bt + kg * 128);
            acc = Q6_Vw_vrmpyacc_VwVbVb(acc, va, vb);
        }

        /* Extract accumulators */
        *(HVX_Vector *)accbuf = acc;

        /* Step B: saturating clamp to int8 and store scores */
        int8_t *srow = scores + i * GN;
        for (int j = 0; j < GN; j++) {
            int32_t v = accbuf[j];
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            srow[j] = (int8_t)v;
        }
    }

    /* Step C: rowwise softmax */
    for (int i = 0; i < GM; i++) {
        const int8_t *row  = scores + i * GN;
        uint8_t      *orow = out    + i * GN;

        /* max */
        int8_t m = row[0];
        for (int j = 1; j < GN; j++)
            if (row[j] > m) m = row[j];

        /* exp_lut lookup + sum */
        int32_t S = 0;
        for (int j = 0; j < GN; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            orow[j] = e;
            S += (int32_t)e;
        }

        /* normalize */
        int32_t half_S = S / 2;
        for (int j = 0; j < GN; j++) {
            int32_t ej = (int32_t)orow[j];
            orow[j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}
