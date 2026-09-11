/*
 * i8_depthwise_conv2d_requant HVX kernel — expert
 *
 * Shapes (from harness): H=W=14, C=8, filter=3x3, stride=1, SAME pad.
 * Depthwise: each channel c convolves ONLY its own input channel with its
 * own 3x3 filter -- NO reduction across channels (unlike standard conv).
 *
 * Defining MAC: Q6_Vh_vmpyi_VhVh on Q6_Wh_vunpack_Vb-widened operands (signed
 * byte x signed byte -> int16, elementwise, NO cross-lane reduction --
 * exactly matches depthwise semantics). NOTE: Q6_Wh_vmpy_VbVb (the more
 * "obvious" byte-multiply intrinsic) DEINTERLEAVES its result (lo=even byte
 * products, hi=odd byte products) -- confirmed by reading m1_driver/tasks/
 * i8_prelu/hvx_v15_vbvb_fix.c's documented investigation, and independently
 * verified here by a failing first attempt (used it, got wrong results).
 * Q6_Wh_vunpack_Vb (sign-extend byte->halfword) instead preserves SEQUENTIAL
 * lane order (lo=lanes 0..63, hi=lanes 64..127), so it composes correctly
 * with plain elementwise multiply.
 *
 * Layout idiom: NHWC row = W*C = 112 bytes. Lane index within a row vector
 * is (x*C + c). For each of the 9 taps (ky,kx) the weight varies only by c
 * (period C=8, same for every x), so we build a weight-broadcast vector:
 * bytes [8*i .. 8*i+7] = wt[c=0..7][ky][kx] for every position i (replicated
 * across all W positions). Elementwise Q6_Wh_vmpy_VbVb against the
 * correspondingly kx/ky-shifted input row gives, per lane, the exact
 * per-channel tap product (int16, safe: |127*127|=16129 << 32767) with NO
 * reduction across c -- this is what makes it depthwise instead of standard
 * conv (which would use vrmpy to REDUCE across c).
 *
 * Row shifting for (ky,kx): padded buffer has (H+2) rows of ROW=144B each
 * (16B left pad + 112B pixel data + 16B right pad). Row (y+1) holds input
 * row y at byte offset C=8. An unaligned vector load at byte offset
 * 8 + kx*8 - 8 = kx*8 within the row selects lane (x*C+c) = pixel
 * (x + kx - 1) channel c -- i.e. exactly the tap-shifted row, matching the
 * conv2d_bias/stride2 vror-offset idiom but via byte-offset load (row width
 * 112B leaves slack for the pad, unlike stride2's exact-128B rows).
 *
 * Accumulation: per tap, widen the int16 product pair to int32 via
 * Q6_Ww_vunpack_Vh and add into running int32 accumulator vectors (4 vectors
 * cover the 112-byte/56-halfword row -> 56 int32 lanes, only ~14*8=112 of the
 * possible 128 lanes are real data but that's fine, extras are pad/garbage
 * and never written to out[]).
 */

#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, const int8_t *wt,
                      int8_t *out,
                      int H, int W, int C,
                      int32_t mult, int shift, int8_t zp) {
    const int PW  = W * C;      /* 112 bytes/row for W=14,C=8 */
    const int ROW = 144;        /* 16B left pad + PW + slack + 16B right pad */

    /* Padded buffer: (H+2) rows x ROW bytes. Row (y+1) holds input row y,
       starting at byte offset C (8B left pad). Rows 0 and H+1 all-zero. */
    int8_t padded[16 * 144] __attribute__((aligned(128)));
    memset(padded, 0, (size_t)(H + 2) * ROW);
    for (int y = 0; y < H; y++)
        memcpy(padded + (y + 1) * ROW + C, in + y * PW, PW);

    /*
     * Weight-broadcast vectors: for each tap t=0..8, a 128-byte vector where
     * every 8-byte group = wt[0..7][ky][kx] (replicated 16x to fill 128B).
     * wt layout is [C][3][3] (per-channel filter), so tap element for
     * channel c is wt[c*9 + t].
     */
    int8_t wt_vecs[9 * 128] __attribute__((aligned(128)));
    for (int t = 0; t < 9; t++) {
        int64_t *dst64 = (int64_t *)(wt_vecs + t * 128);
        int64_t wp = 0;
        for (int c = 0; c < C; c++)
            wp |= ((int64_t)(uint8_t)wt[c * 9 + t]) << (8 * c);
        for (int i = 0; i < 16; i++) dst64[i] = wp;
    }

    const int32_t half = (shift > 0) ? (1 << (shift - 1)) : 0;
    const int32_t izp  = (int32_t)zp;

    HVX_Vector vw0 = *(const HVX_Vector *)(wt_vecs + 0*128);
    HVX_Vector vw1 = *(const HVX_Vector *)(wt_vecs + 1*128);
    HVX_Vector vw2 = *(const HVX_Vector *)(wt_vecs + 2*128);
    HVX_Vector vw3 = *(const HVX_Vector *)(wt_vecs + 3*128);
    HVX_Vector vw4 = *(const HVX_Vector *)(wt_vecs + 4*128);
    HVX_Vector vw5 = *(const HVX_Vector *)(wt_vecs + 5*128);
    HVX_Vector vw6 = *(const HVX_Vector *)(wt_vecs + 6*128);
    HVX_Vector vw7 = *(const HVX_Vector *)(wt_vecs + 7*128);
    HVX_Vector vw8 = *(const HVX_Vector *)(wt_vecs + 8*128);
    HVX_Vector wtap_arr[9] = {vw0,vw1,vw2,vw3,vw4,vw5,vw6,vw7,vw8};

    /* Widen weight taps to int16 ONCE (loop-invariant across y). */
    HVX_Vector wlo_arr[9], whi_arr[9];
    for (int t = 0; t < 9; t++) {
        HVX_VectorPair wtw = Q6_Wh_vunpack_Vb(wtap_arr[t]);
        wlo_arr[t] = Q6_V_lo_W(wtw);
        whi_arr[t] = Q6_V_hi_W(wtw);
    }

    /* acc32 staging: 128 int32 lanes (4 vectors) -- lane (x*C+c) low 112 used. */
    int32_t accbuf[128] __attribute__((aligned(128)));

    for (int y = 0; y < H; y++) {
        const int8_t *r0 = padded + (y + 0) * ROW; /* input row y-1 */
        const int8_t *r1 = padded + (y + 1) * ROW; /* input row y   */
        const int8_t *r2 = padded + (y + 2) * ROW; /* input row y+1 */

        /* kx=0,1,2 -> byte offsets 0,8,16 within the row (lane i -> pixel
           (i/C + kx - 1), channel i%C). */
        HVX_Vector v0_k0 = *(const HVX_UVector *)(r0 + 0);
        HVX_Vector v0_k1 = *(const HVX_UVector *)(r0 + 8);
        HVX_Vector v0_k2 = *(const HVX_UVector *)(r0 + 16);
        HVX_Vector v1_k0 = *(const HVX_UVector *)(r1 + 0);
        HVX_Vector v1_k1 = *(const HVX_UVector *)(r1 + 8);
        HVX_Vector v1_k2 = *(const HVX_UVector *)(r1 + 16);
        HVX_Vector v2_k0 = *(const HVX_UVector *)(r2 + 0);
        HVX_Vector v2_k1 = *(const HVX_UVector *)(r2 + 8);
        HVX_Vector v2_k2 = *(const HVX_UVector *)(r2 + 16);

        /* Widen each input row (byte) to int16, SEQUENTIAL lane order
           (lo=lanes 0..63, hi=lanes 64..127) via Q6_Wh_vunpack_Vb. Weight
           vectors are widened once (they don't depend on y). Elementwise
           Q6_Vh_vmpyi_VhVh multiply per half (NO cross-lane reduction --
           depthwise). Each tap -> 2 halfword-vector products, safe in int16
           (|127*127|=16129 < 32767, single product per lane, no summing yet). */
        HVX_VectorPair i0 = Q6_Wh_vunpack_Vb(v0_k0);
        HVX_VectorPair i1 = Q6_Wh_vunpack_Vb(v0_k1);
        HVX_VectorPair i2 = Q6_Wh_vunpack_Vb(v0_k2);
        HVX_VectorPair i3 = Q6_Wh_vunpack_Vb(v1_k0);
        HVX_VectorPair i4 = Q6_Wh_vunpack_Vb(v1_k1);
        HVX_VectorPair i5 = Q6_Wh_vunpack_Vb(v1_k2);
        HVX_VectorPair i6 = Q6_Wh_vunpack_Vb(v2_k0);
        HVX_VectorPair i7 = Q6_Wh_vunpack_Vb(v2_k1);
        HVX_VectorPair i8 = Q6_Wh_vunpack_Vb(v2_k2);
        HVX_VectorPair iw[9] = {i0,i1,i2,i3,i4,i5,i6,i7,i8};

        /* Sum the 9 taps' products in int32 (widen BEFORE summing to avoid
           any wraparound: 9 * 16129 = 145161 exceeds int16 range). */
        HVX_Vector acc_w0 = Q6_V_vzero(); /* byte-lanes  0..31 */
        HVX_Vector acc_w1 = Q6_V_vzero(); /* byte-lanes 32..63 */
        HVX_Vector acc_w2 = Q6_V_vzero(); /* byte-lanes 64..95 */
        HVX_Vector acc_w3 = Q6_V_vzero(); /* byte-lanes 96..127 */
        for (int t = 0; t < 9; t++) {
            HVX_Vector ilo = Q6_V_lo_W(iw[t]);           /* int16 lanes 0..63 */
            HVX_Vector ihi = Q6_V_hi_W(iw[t]);           /* int16 lanes 64..127 */
            HVX_Vector plo = Q6_Vh_vmpyi_VhVh(ilo, wlo_arr[t]); /* elementwise, int16 */
            HVX_Vector phi = Q6_Vh_vmpyi_VhVh(ihi, whi_arr[t]);
            HVX_VectorPair wplo = Q6_Ww_vunpack_Vh(plo); /* widen to int32, sequential */
            HVX_VectorPair wphi = Q6_Ww_vunpack_Vh(phi);
            acc_w0 = Q6_Vw_vadd_VwVw(acc_w0, Q6_V_lo_W(wplo));
            acc_w1 = Q6_Vw_vadd_VwVw(acc_w1, Q6_V_hi_W(wplo));
            acc_w2 = Q6_Vw_vadd_VwVw(acc_w2, Q6_V_lo_W(wphi));
            acc_w3 = Q6_Vw_vadd_VwVw(acc_w3, Q6_V_hi_W(wphi));
        }

        *(HVX_Vector *)(accbuf +  0) = acc_w0; /* byte-lanes  0..31 -> int32 idx 0..31 */
        *(HVX_Vector *)(accbuf + 32) = acc_w1; /* byte-lanes 32..63 */
        *(HVX_Vector *)(accbuf + 64) = acc_w2; /* byte-lanes 64..95 */
        *(HVX_Vector *)(accbuf + 96) = acc_w3; /* byte-lanes 96..127 */

        /* accbuf[i] corresponds to byte-lane i of the tap-shifted rows, i.e.
           pixel (i/C), channel (i%C), for output row y. */
        int8_t *out_row = out + y * PW;
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C; c++) {
                int32_t v = accbuf[x * C + c] * mult;
                int32_t r;
                if (v >= 0) r = (v + half) >> shift;
                else        r = -((-v + half) >> shift);
                r += izp;
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                out_row[x * C + c] = (int8_t)r;
            }
        }
    }
}
