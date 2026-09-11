/*
 * HVX kernel: i8_token_mix_block
 * MLP-Mixer token-mixing block: 2-layer GEMM over tokens.
 *
 * Dims: TOKENS=8, CHANNELS=16, D_MIX=16
 *
 * Strategy:
 *   Process all 16 channels simultaneously using HVX word-lane accumulators.
 *   Pre-transpose X: XT_g[group][c] = 4 bytes of X for tokens in that group, channel c.
 *   Pre-transpose y1 similarly for step 2.
 *
 *   vrmpy: acc[lane] += dot4(w_splatted[lane], x_transposed[lane])
 *   W group bytes splat into all 32 word lanes; XT has 16 valid word lanes (channels 0..15).
 *   Requant in scalar int64 for exact bit-match.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

#define TOKENS   8
#define CHANNELS 16
#define D_MIX    16

static inline int8_t requant_scalar(int32_t acc, int32_t bias_v, int32_t mult, int shift) {
    int64_t biased = (int64_t)acc + (int64_t)bias_v;
    int64_t v      = biased * (int64_t)mult;
    int64_t half   = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q      = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

void candidate_kernel(const int8_t  *X,
                      const int8_t  *W,
                      const int32_t *b,
                      const int8_t  *W2,
                      const int32_t *b2,
                      int8_t        *out,
                      int32_t mult1, int shift1,
                      int32_t mult2, int shift2)
{
    /*
     * Transpose X into two 128-byte-aligned groups of 4 tokens each.
     * XT_g0[c*4 + q] = X[(0+q)*CHANNELS + c]  for q=0..3, c=0..15 (t=0..3)
     * XT_g1[c*4 + q] = X[(4+q)*CHANNELS + c]  for q=0..3, c=0..15 (t=4..7)
     * Each group is 64 valid bytes; padded to 128 for aligned HVX load.
     */
    int8_t __attribute__((aligned(128))) XT_g0[128]; /* 128 bytes, lanes 0..15 valid */
    int8_t __attribute__((aligned(128))) XT_g1[128];

    memset(XT_g0, 0, 128);
    memset(XT_g1, 0, 128);

    for (int c = 0; c < CHANNELS; c++) {
        XT_g0[c*4+0] = X[0*CHANNELS + c];
        XT_g0[c*4+1] = X[1*CHANNELS + c];
        XT_g0[c*4+2] = X[2*CHANNELS + c];
        XT_g0[c*4+3] = X[3*CHANNELS + c];

        XT_g1[c*4+0] = X[4*CHANNELS + c];
        XT_g1[c*4+1] = X[5*CHANNELS + c];
        XT_g1[c*4+2] = X[6*CHANNELS + c];
        XT_g1[c*4+3] = X[7*CHANNELS + c];
    }

    HVX_Vector vXT_g0 = *(const HVX_Vector *)XT_g0;
    HVX_Vector vXT_g1 = *(const HVX_Vector *)XT_g1;

    /*
     * Step 1: For each mixer row m (0..15), compute y1[m][c] for all c=0..15.
     * y1 stored as [D_MIX x CHANNELS]: y1_flat[m*CHANNELS + c].
     */
    int8_t __attribute__((aligned(128))) y1[D_MIX * CHANNELS]; /* 256 bytes */
    int32_t __attribute__((aligned(128))) accbuf[32]; /* 128 bytes = 1 HVX vector */

    for (int m = 0; m < D_MIX; m++) {
        /* Splat W[m, t=0..3] and W[m, t=4..7] across all word lanes */
        int32_t w4_0 = (int32_t)((uint8_t)W[m*TOKENS + 0])
                     | (int32_t)((uint8_t)W[m*TOKENS + 1] << 8)
                     | (int32_t)((uint8_t)W[m*TOKENS + 2] << 16)
                     | (int32_t)((uint8_t)W[m*TOKENS + 3] << 24);
        int32_t w4_1 = (int32_t)((uint8_t)W[m*TOKENS + 4])
                     | (int32_t)((uint8_t)W[m*TOKENS + 5] << 8)
                     | (int32_t)((uint8_t)W[m*TOKENS + 6] << 16)
                     | (int32_t)((uint8_t)W[m*TOKENS + 7] << 24);

        HVX_Vector va0 = Q6_V_vsplat_R(w4_0);
        HVX_Vector va1 = Q6_V_vsplat_R(w4_1);

        /* acc[lane c] = sum_{q=0}^{3} W[m,q]*X[q,c] + W[m,4+q]*X[4+q,c]  for c=0..15 */
        HVX_Vector vacc = Q6_V_vzero();
        vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, va0, vXT_g0);
        vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, va1, vXT_g1);

        *(HVX_Vector *)accbuf = vacc;

        for (int c = 0; c < CHANNELS; c++) {
            y1[m * CHANNELS + c] = requant_scalar(accbuf[c], b[m], mult1, shift1);
        }
    }

    /*
     * Step 2: For each output token t (0..7), compute out[t,c] for all c=0..15.
     * D_MIX=16 -> 4 groups of 4.
     * Transpose y1: Y1T_g[g][c*4 + q] = y1[(g*4+q)*CHANNELS + c]
     * Each group 64 valid bytes, padded to 128.
     */
    int8_t __attribute__((aligned(128))) Y1T_g0[128];
    int8_t __attribute__((aligned(128))) Y1T_g1[128];
    int8_t __attribute__((aligned(128))) Y1T_g2[128];
    int8_t __attribute__((aligned(128))) Y1T_g3[128];

    memset(Y1T_g0, 0, 128);
    memset(Y1T_g1, 0, 128);
    memset(Y1T_g2, 0, 128);
    memset(Y1T_g3, 0, 128);

    for (int c = 0; c < CHANNELS; c++) {
        Y1T_g0[c*4+0] = y1[( 0)*CHANNELS + c];
        Y1T_g0[c*4+1] = y1[( 1)*CHANNELS + c];
        Y1T_g0[c*4+2] = y1[( 2)*CHANNELS + c];
        Y1T_g0[c*4+3] = y1[( 3)*CHANNELS + c];

        Y1T_g1[c*4+0] = y1[( 4)*CHANNELS + c];
        Y1T_g1[c*4+1] = y1[( 5)*CHANNELS + c];
        Y1T_g1[c*4+2] = y1[( 6)*CHANNELS + c];
        Y1T_g1[c*4+3] = y1[( 7)*CHANNELS + c];

        Y1T_g2[c*4+0] = y1[( 8)*CHANNELS + c];
        Y1T_g2[c*4+1] = y1[( 9)*CHANNELS + c];
        Y1T_g2[c*4+2] = y1[(10)*CHANNELS + c];
        Y1T_g2[c*4+3] = y1[(11)*CHANNELS + c];

        Y1T_g3[c*4+0] = y1[(12)*CHANNELS + c];
        Y1T_g3[c*4+1] = y1[(13)*CHANNELS + c];
        Y1T_g3[c*4+2] = y1[(14)*CHANNELS + c];
        Y1T_g3[c*4+3] = y1[(15)*CHANNELS + c];
    }

    HVX_Vector vY1T_g0 = *(const HVX_Vector *)Y1T_g0;
    HVX_Vector vY1T_g1 = *(const HVX_Vector *)Y1T_g1;
    HVX_Vector vY1T_g2 = *(const HVX_Vector *)Y1T_g2;
    HVX_Vector vY1T_g3 = *(const HVX_Vector *)Y1T_g3;

    for (int t = 0; t < TOKENS; t++) {
        int32_t w2_0 = (int32_t)((uint8_t)W2[t*D_MIX +  0])
                     | (int32_t)((uint8_t)W2[t*D_MIX +  1] << 8)
                     | (int32_t)((uint8_t)W2[t*D_MIX +  2] << 16)
                     | (int32_t)((uint8_t)W2[t*D_MIX +  3] << 24);
        int32_t w2_1 = (int32_t)((uint8_t)W2[t*D_MIX +  4])
                     | (int32_t)((uint8_t)W2[t*D_MIX +  5] << 8)
                     | (int32_t)((uint8_t)W2[t*D_MIX +  6] << 16)
                     | (int32_t)((uint8_t)W2[t*D_MIX +  7] << 24);
        int32_t w2_2 = (int32_t)((uint8_t)W2[t*D_MIX +  8])
                     | (int32_t)((uint8_t)W2[t*D_MIX +  9] << 8)
                     | (int32_t)((uint8_t)W2[t*D_MIX + 10] << 16)
                     | (int32_t)((uint8_t)W2[t*D_MIX + 11] << 24);
        int32_t w2_3 = (int32_t)((uint8_t)W2[t*D_MIX + 12])
                     | (int32_t)((uint8_t)W2[t*D_MIX + 13] << 8)
                     | (int32_t)((uint8_t)W2[t*D_MIX + 14] << 16)
                     | (int32_t)((uint8_t)W2[t*D_MIX + 15] << 24);

        HVX_Vector vw0 = Q6_V_vsplat_R(w2_0);
        HVX_Vector vw1 = Q6_V_vsplat_R(w2_1);
        HVX_Vector vw2 = Q6_V_vsplat_R(w2_2);
        HVX_Vector vw3 = Q6_V_vsplat_R(w2_3);

        HVX_Vector vacc2 = Q6_V_vzero();
        vacc2 = Q6_Vw_vrmpyacc_VwVbVb(vacc2, vw0, vY1T_g0);
        vacc2 = Q6_Vw_vrmpyacc_VwVbVb(vacc2, vw1, vY1T_g1);
        vacc2 = Q6_Vw_vrmpyacc_VwVbVb(vacc2, vw2, vY1T_g2);
        vacc2 = Q6_Vw_vrmpyacc_VwVbVb(vacc2, vw3, vY1T_g3);

        *(HVX_Vector *)accbuf = vacc2;

        for (int c = 0; c < CHANNELS; c++) {
            out[t * CHANNELS + c] = requant_scalar(accbuf[c], b2[t], mult2, shift2);
        }
    }
}
