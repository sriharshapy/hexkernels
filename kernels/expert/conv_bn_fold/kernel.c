/*
 * conv_bn_fold HVX kernel — expert
 *
 * Shapes (from harness): H=8, W=8, C_in=8, C_out=8, filter=3x3, stride=1, SAME pad.
 * Identical shape/layout family to i8_conv2d_bias, differing only in the
 * requantize step: PER-CHANNEL bn_scale[co]/bn_shift[co] (not a single global
 * mult/shift) plus a folded per-channel bias[co] and a global zero-point zp.
 *
 * Defining MAC: Q6_Vw_vrmpyacc_VwVbVb
 *   Signed int8*int8 reduce-multiply accumulate: each int32 lane i accumulates
 *   a[4i]*b[4i] + a[4i+1]*b[4i+1] + a[4i+2]*b[4i+2] + a[4i+3]*b[4i+3].
 *   4 signed MACs per int32 lane, 32 lanes per 128B vector = 128 MACs/cycle.
 *
 * Conv mapping (same idiom as i8_conv2d_bias):
 *   Input layout: zero-padded rows at 128-byte stride (W*C_in=64 pixel bytes +
 *   8B left pad @ offset 0..7 + pixels @ 8..71 + zero tail to 128).
 *   Weight layout: for each (co, tap), 8-byte weight [w0..w7] replicated across
 *   all W=8 pixel positions (64 bytes), then 64 zero bytes = 128B vector.
 *   After vrmpyacc, lane 2x = dot4(in[x][ci=0..3], w[ci=0..3]) and
 *   lane 2x+1 = dot4(in[x][ci=4..7], w[ci=4..7]). Pair-sum gives acc[x].
 *   kx offset handled by vror(input_row, kx*8) shifting pixel positions.
 *
 * 3-way independent accumulator chains (va/vb/vc per kx=0/1/2) hide the
 * 4-cycle vrmpyacc latency. All 9 weight vectors preloaded before the chain.
 *
 * Pair-sum: vror(vacc,4) + vacc -> accbuf[2x] = acc[x].
 * Tail: scalar per-channel requantize (bias/scale/shift/zp hoisted out of the
 * pixel loop, computed once per (co,y)) with round-half-away-from-zero exactly
 * matching baseline.c's int64 formula (values fit in int64/int32 safely since
 * |acc| <= 8*9*127*127 ~= 1.16M and bn_scale/bias are bounded in the harness).
 */

#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, const int8_t *wt,
                      const int32_t *bias,
                      const int32_t *bn_scale, const int *bn_shift,
                      int8_t zp,
                      int8_t *out,
                      int H, int W, int C_in, int C_out) {
    const int PHR = 128;

    /*
     * Padded input buffer: (H+2) rows x 128-byte stride.
     * Row y occupies bytes [(y+1)*128 .. (y+1)*128+127].
     * Pixels start at offset C_in=8 (left zero-pad column).
     * Top and bottom rows stay zero (top/bottom padding).
     */
    int8_t padded[12 * 128] __attribute__((aligned(128)));
    memset(padded, 0, (size_t)(H + 2) * PHR);
    for (int y = 0; y < H; y++)
        memcpy(padded + (y + 1) * PHR + C_in, in + y * W * C_in, (size_t)W * C_in);

    /*
     * Weight vectors: for each (co, tap t=0..8), a 128-byte vector.
     * Bytes 0..63 = 8-byte weight pattern [w0..w7] replicated 8 times (W=8 pixels).
     * Bytes 64..127 = zero (lanes 16..31 contribute 0 to accumulators).
     */
    int8_t wt_vecs[8 * 9 * 128] __attribute__((aligned(128)));
    for (int co = 0; co < C_out; co++) {
        for (int t = 0; t < 9; t++) {
            const int8_t *wtp = wt + co * 9 * C_in + t * C_in;
            int64_t *dst64 = (int64_t *)(wt_vecs + (co * 9 + t) * 128);
            int64_t wp =
                ((int64_t)(uint8_t)wtp[0])        |
                ((int64_t)(uint8_t)wtp[1] <<  8)  |
                ((int64_t)(uint8_t)wtp[2] << 16)  |
                ((int64_t)(uint8_t)wtp[3] << 24)  |
                ((int64_t)(uint8_t)wtp[4] << 32)  |
                ((int64_t)(uint8_t)wtp[5] << 40)  |
                ((int64_t)(uint8_t)wtp[6] << 48)  |
                ((int64_t)(uint8_t)wtp[7] << 56);
            dst64[0]=wp; dst64[1]=wp; dst64[2]=wp; dst64[3]=wp;
            dst64[4]=wp; dst64[5]=wp; dst64[6]=wp; dst64[7]=wp;
            dst64[ 8]=0; dst64[ 9]=0; dst64[10]=0; dst64[11]=0;
            dst64[12]=0; dst64[13]=0; dst64[14]=0; dst64[15]=0;
        }
    }

    const int32_t izp = (int32_t)zp;

    /* Accumulator staging buffer for pair-sum readout */
    int32_t accbuf[32] __attribute__((aligned(128)));

    for (int co = 0; co < C_out; co++) {
        const int8_t *cw = wt_vecs + co * 9 * 128;
        const int32_t bco   = bias[co];
        const int32_t scale = bn_scale[co];
        const int     shift = bn_shift[co];
        const int64_t half  = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;

        for (int y = 0; y < H; y++) {
            /* Load 3 input rows for the 3x3 kernel window */
            HVX_Vector v0 = *(const HVX_Vector *)(padded + (y+0)*PHR);
            HVX_Vector v1 = *(const HVX_Vector *)(padded + (y+1)*PHR);
            HVX_Vector v2 = *(const HVX_Vector *)(padded + (y+2)*PHR);

            /* kx rotations: vror by kx*C_in bytes shifts pixel window right by kx */
            HVX_Vector v0r8  = Q6_V_vror_VR(v0,  8);
            HVX_Vector v0r16 = Q6_V_vror_VR(v0, 16);
            HVX_Vector v1r8  = Q6_V_vror_VR(v1,  8);
            HVX_Vector v1r16 = Q6_V_vror_VR(v1, 16);
            HVX_Vector v2r8  = Q6_V_vror_VR(v2,  8);
            HVX_Vector v2r16 = Q6_V_vror_VR(v2, 16);

            /* Preload all 9 weight vectors for this (co, y) */
            HVX_Vector vw0 = *(const HVX_Vector *)(cw + 0*128);
            HVX_Vector vw1 = *(const HVX_Vector *)(cw + 1*128);
            HVX_Vector vw2 = *(const HVX_Vector *)(cw + 2*128);
            HVX_Vector vw3 = *(const HVX_Vector *)(cw + 3*128);
            HVX_Vector vw4 = *(const HVX_Vector *)(cw + 4*128);
            HVX_Vector vw5 = *(const HVX_Vector *)(cw + 5*128);
            HVX_Vector vw6 = *(const HVX_Vector *)(cw + 6*128);
            HVX_Vector vw7 = *(const HVX_Vector *)(cw + 7*128);
            HVX_Vector vw8 = *(const HVX_Vector *)(cw + 8*128);

            /* 3-way independent vrmpyacc chains to hide 4-cycle latency
             * (chain va: kx=0 taps 0,3,6; vb: kx=1 taps 1,4,7; vc: kx=2 taps 2,5,8) */
            HVX_Vector va = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), v0,    vw0);
            HVX_Vector vb = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), v0r8,  vw1);
            HVX_Vector vc = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), v0r16, vw2);

            va = Q6_Vw_vrmpyacc_VwVbVb(va, v1,    vw3);
            vb = Q6_Vw_vrmpyacc_VwVbVb(vb, v1r8,  vw4);
            vc = Q6_Vw_vrmpyacc_VwVbVb(vc, v1r16, vw5);

            va = Q6_Vw_vrmpyacc_VwVbVb(va, v2,    vw6);
            vb = Q6_Vw_vrmpyacc_VwVbVb(vb, v2r8,  vw7);
            vc = Q6_Vw_vrmpyacc_VwVbVb(vc, v2r16, vw8);

            HVX_Vector vacc = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(va, vb), vc);

            /* Pair-sum: rotate by 4 bytes (1 int32), add -> accbuf[2x] = acc[x] */
            HVX_Vector vrot = Q6_V_vror_VR(vacc, 4);
            *(HVX_Vector *)accbuf = Q6_Vw_vadd_VwVw(vacc, vrot);

            /* Scalar per-channel requantize, W=8 pixels per (co, y). Matches
             * baseline.c's int64 round-half-away-from-zero exactly. */
            int8_t *out_row = out + y * W * C_out;
            for (int x = 0; x < W; x++) {
                int64_t biased = (int64_t)accbuf[2*x] + (int64_t)bco;
                int64_t v      = biased * (int64_t)scale;
                int64_t r      = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
                r += izp;
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                out_row[x * C_out + co] = (int8_t)r;
            }
        }
    }
}
