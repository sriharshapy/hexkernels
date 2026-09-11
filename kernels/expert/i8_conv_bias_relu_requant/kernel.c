/*
 * i8_conv_bias_relu_requant -- HVX candidate kernel
 *
 * Task: 3x3 SAME-pad conv2d (int8) + bias (int32) + ReLU + requantize -> int8
 * Shape (pinned by harness): H=8, W=8, C_in=8, C_out=4
 *
 * HVX idiom (same as i8_conv2d_bias winning approach):
 *  1. Zero-pad input into aligned 128-byte rows (SAME padding = 1 pixel border).
 *  2. Replicate each weight tap (8 bytes) across a full 128-byte vector.
 *  3. For each output channel + each output row, accumulate 9 taps using
 *     Q6_Vw_vrmpyacc_VwVbVb on the shifted input rows (Q6_V_vror_VR for col shifts).
 *     Each output pixel x occupies 2 consecutive int32 lanes (C_in=8 = 2*4).
 *     Fold them: accbuf[2*x] holds the sum after vror(vacc,4)+vadd.
 *  4. Scalar epilogue per pixel: add bias, ReLU (clamp <0 to 0 in int64 domain),
 *     requantize (mult/shift/round-half-away-from-zero), add zp, saturate to int8.
 */

#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, const int8_t *wt, const int32_t *bias,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp) {
    /* ---- 1. Build padded input rows (zero-pad border = 1 pixel = C_in bytes) ---- */
    /* PHR = 128 bytes per padded row; H+2 rows (top zero + H rows + bottom zero) */
    const int PHR = 128;
    int8_t padded[(8 + 2) * 128] __attribute__((aligned(128)));
    memset(padded, 0, (H + 2) * PHR);
    for (int y = 0; y < H; y++)
        /* offset C_in bytes in to leave one pixel of left zero-padding */
        memcpy(padded + (y + 1) * PHR + C_in, in + y * W * C_in, W * C_in);

    /* ---- 2. Replicate weight vectors ---- */
    /* wt layout: [C_out][3][3][C_in] = [4][9][8].
     * For each (co, tap), broadcast the 8-byte weight slice across 128 bytes. */
    int8_t wt_vecs[4 * 9 * 128] __attribute__((aligned(128)));
    for (int co = 0; co < C_out; co++) {
        for (int t = 0; t < 9; t++) {
            const int8_t *wtp = wt + co * 9 * C_in + t * C_in;
            int64_t *dst64 = (int64_t *)(wt_vecs + (co * 9 + t) * 128);
            /* Pack 8 weight bytes into one int64, then replicate 16 times */
            int64_t wp =
                ((int64_t)(uint8_t)wtp[0])        |
                ((int64_t)(uint8_t)wtp[1] <<  8)  |
                ((int64_t)(uint8_t)wtp[2] << 16)  |
                ((int64_t)(uint8_t)wtp[3] << 24)  |
                ((int64_t)(uint8_t)wtp[4] << 32)  |
                ((int64_t)(uint8_t)wtp[5] << 40)  |
                ((int64_t)(uint8_t)wtp[6] << 48)  |
                ((int64_t)(uint8_t)wtp[7] << 56);
            /* W=8 pixels active; fill all 16 slots (tail stays 0 from weight) */
            dst64[ 0]=wp; dst64[ 1]=wp; dst64[ 2]=wp; dst64[ 3]=wp;
            dst64[ 4]=wp; dst64[ 5]=wp; dst64[ 6]=wp; dst64[ 7]=wp;
            dst64[ 8]=wp; dst64[ 9]=wp; dst64[10]=wp; dst64[11]=wp;
            dst64[12]=wp; dst64[13]=wp; dst64[14]=wp; dst64[15]=wp;
        }
    }

    /* ---- 3. Hoist requant constants ---- */
    int32_t half = (shift > 0) ? (1 << (shift - 1)) : 0;
    int32_t izp  = (int32_t)zp;

    /* accbuf: 32 int32 lanes from one 128-byte HVX vector */
    int32_t accbuf[32] __attribute__((aligned(128)));

    /* ---- 4. Main loop: for each output channel, for each row ---- */
    for (int co = 0; co < C_out; co++) {
        const int8_t *cw  = wt_vecs + co * 9 * 128;
        int32_t       bco = bias[co];

        /* Preload weight vectors for all 9 taps */
        HVX_Vector vw0 = *(const HVX_Vector *)(cw + 0*128);
        HVX_Vector vw1 = *(const HVX_Vector *)(cw + 1*128);
        HVX_Vector vw2 = *(const HVX_Vector *)(cw + 2*128);
        HVX_Vector vw3 = *(const HVX_Vector *)(cw + 3*128);
        HVX_Vector vw4 = *(const HVX_Vector *)(cw + 4*128);
        HVX_Vector vw5 = *(const HVX_Vector *)(cw + 5*128);
        HVX_Vector vw6 = *(const HVX_Vector *)(cw + 6*128);
        HVX_Vector vw7 = *(const HVX_Vector *)(cw + 7*128);
        HVX_Vector vw8 = *(const HVX_Vector *)(cw + 8*128);

        for (int y = 0; y < H; y++) {
            /* Load 3 input rows (with zero-pad border) */
            HVX_Vector v0 = *(const HVX_Vector *)(padded + (y+0)*PHR);
            HVX_Vector v1 = *(const HVX_Vector *)(padded + (y+1)*PHR);
            HVX_Vector v2 = *(const HVX_Vector *)(padded + (y+2)*PHR);

            /* Column-shifted versions (8 bytes = C_in bytes = one pixel shift) */
            HVX_Vector v0r8  = Q6_V_vror_VR(v0,  8);
            HVX_Vector v0r16 = Q6_V_vror_VR(v0, 16);
            HVX_Vector v1r8  = Q6_V_vror_VR(v1,  8);
            HVX_Vector v1r16 = Q6_V_vror_VR(v1, 16);
            HVX_Vector v2r8  = Q6_V_vror_VR(v2,  8);
            HVX_Vector v2r16 = Q6_V_vror_VR(v2, 16);

            /* Accumulate 9 taps into 3 partial accumulators for ILP */
            HVX_Vector vacc0 = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), v0,    vw0);
            HVX_Vector vacc1 = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), v0r8,  vw1);
            HVX_Vector vacc2 = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), v0r16, vw2);

            vacc0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0, v1,    vw3);
            vacc1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1, v1r8,  vw4);
            vacc2 = Q6_Vw_vrmpyacc_VwVbVb(vacc2, v1r16, vw5);

            vacc0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0, v2,    vw6);
            vacc1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1, v2r8,  vw7);
            vacc2 = Q6_Vw_vrmpyacc_VwVbVb(vacc2, v2r16, vw8);

            /* Fold partial sums: vacc = vacc0 + vacc1 + vacc2 */
            HVX_Vector vacc = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(vacc0, vacc1), vacc2);

            /* With C_in=8, each output pixel uses 2 consecutive int32 lanes.
             * Lane 2*x holds dot-product of bytes [8*x..8*x+3] with wt[0..3].
             * Lane 2*x+1 holds dot-product of bytes [8*x+4..8*x+7] with wt[4..7].
             * Fold by rotating 4 bytes (one int32) and adding to sum upper+lower halves. */
            HVX_Vector vrot = Q6_V_vror_VR(vacc, 4);
            *(HVX_Vector *)accbuf = Q6_Vw_vadd_VwVw(vacc, vrot);

            /* ---- Scalar epilogue: bias + ReLU + requant ---- */
            int8_t *out_row = out + y * W * C_out;
            for (int x = 0; x < W; x++) {
                /* accbuf[2*x] contains the full conv sum for pixel (y,x,co) */
                int64_t biased = (int64_t)accbuf[2*x] + (int64_t)bco;
                /* ReLU */
                if (biased < 0) biased = 0;
                /* Requantize: round-half-away-from-zero */
                int64_t v = biased * (int64_t)mult;
                int64_t r;
                if (v >= 0) r = (v + (int64_t)half) >> shift;
                else        r = -((-v + (int64_t)half) >> shift);
                r += izp;
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                out_row[x * C_out + co] = (int8_t)r;
            }
        }
    }
}
