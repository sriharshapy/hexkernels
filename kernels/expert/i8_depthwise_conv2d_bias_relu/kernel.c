/*
 * i8_depthwise_conv2d_bias_relu HVX kernel — expert
 *
 * Shapes: H=W=14, C=8, filter=3x3, stride=1, SAME pad.
 * Depthwise: each channel c convolves ONLY its own input channel with its
 * own 3x3 filter -- NO reduction across channels. Adds per-channel bias,
 * ReLU-clamps (int32), then requantizes to int8.
 *
 * Idiom (proven, see i8_depthwise_conv2d_requant.c): vectorize across the
 * NHWC row lane index (x*C+c). Build per-tap weight-broadcast vectors
 * (period C, replicated across W). Widen input bytes with Q6_Wh_vunpack_Vb
 * (SEQUENTIAL lane order -- do NOT use Q6_Wh_vmpy_VbVb, which deinterleaves
 * lo=even/hi=odd). Multiply elementwise with Q6_Vh_vmpyi_VhVh (int16, safe:
 * |127*127|=16129). Widen to int32 via Q6_Ww_vunpack_Vh before summing the
 * 9 taps (9*16129=145161 overflows int16). Then add per-channel bias
 * (broadcast similarly), ReLU-clamp, requantize.
 */

#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, const int8_t *wt, const int32_t *bias,
                      int8_t *out,
                      int H, int W, int C,
                      int32_t mult, int shift, int8_t zp) {
    const int PW  = W * C;      /* 112 bytes/row for W=14,C=8 */
    const int ROW = 144;        /* 16B left pad + PW + slack + 16B right pad */

    int8_t padded[16 * 144] __attribute__((aligned(128)));
    memset(padded, 0, (size_t)(H + 2) * ROW);
    for (int y = 0; y < H; y++)
        memcpy(padded + (y + 1) * ROW + C, in + y * PW, PW);

    /* Weight-broadcast vectors: tap t -> 128B vector, byte group [8*i..8*i+7]
       = wt[c=0..7][ky][kx] replicated across all W positions. */
    int8_t wt_vecs[9 * 128] __attribute__((aligned(128)));
    for (int t = 0; t < 9; t++) {
        int64_t *dst64 = (int64_t *)(wt_vecs + t * 128);
        int64_t wp = 0;
        for (int c = 0; c < C; c++)
            wp |= ((int64_t)(uint8_t)wt[c * 9 + t]) << (8 * c);
        for (int i = 0; i < 16; i++) dst64[i] = wp;
    }

    /* Per-channel int32 bias, broadcast across x (period C, replicated). */
    int32_t bias_buf[32] __attribute__((aligned(128))); /* 32 int32 lanes = 128B */
    for (int i = 0; i < 32; i++) bias_buf[i] = bias[i % C];
    HVX_Vector vbias0 = *(const HVX_Vector *)(bias_buf); /* lanes 0..31 -> byte lanes 0..31 */
    /* Need 4 such vectors to cover 128 byte-lanes (32 int32 lanes each). All
       identical pattern since period C=8 divides 32. */
    HVX_Vector vbias1 = vbias0, vbias2 = vbias0, vbias3 = vbias0;

    const int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    const int32_t izp  = (int32_t)zp;

    HVX_Vector wtap_arr[9];
    for (int t = 0; t < 9; t++)
        wtap_arr[t] = *(const HVX_Vector *)(wt_vecs + t * 128);

    HVX_Vector wlo_arr[9], whi_arr[9];
    for (int t = 0; t < 9; t++) {
        HVX_VectorPair wtw = Q6_Wh_vunpack_Vb(wtap_arr[t]);
        wlo_arr[t] = Q6_V_lo_W(wtw);
        whi_arr[t] = Q6_V_hi_W(wtw);
    }

    int32_t accbuf[128] __attribute__((aligned(128)));

    for (int y = 0; y < H; y++) {
        const int8_t *r0 = padded + (y + 0) * ROW;
        const int8_t *r1 = padded + (y + 1) * ROW;
        const int8_t *r2 = padded + (y + 2) * ROW;

        HVX_Vector v0_k0 = *(const HVX_UVector *)(r0 + 0);
        HVX_Vector v0_k1 = *(const HVX_UVector *)(r0 + 8);
        HVX_Vector v0_k2 = *(const HVX_UVector *)(r0 + 16);
        HVX_Vector v1_k0 = *(const HVX_UVector *)(r1 + 0);
        HVX_Vector v1_k1 = *(const HVX_UVector *)(r1 + 8);
        HVX_Vector v1_k2 = *(const HVX_UVector *)(r1 + 16);
        HVX_Vector v2_k0 = *(const HVX_UVector *)(r2 + 0);
        HVX_Vector v2_k1 = *(const HVX_UVector *)(r2 + 8);
        HVX_Vector v2_k2 = *(const HVX_UVector *)(r2 + 16);

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

        HVX_Vector acc_w0 = Q6_V_vzero();
        HVX_Vector acc_w1 = Q6_V_vzero();
        HVX_Vector acc_w2 = Q6_V_vzero();
        HVX_Vector acc_w3 = Q6_V_vzero();
        for (int t = 0; t < 9; t++) {
            HVX_Vector ilo = Q6_V_lo_W(iw[t]);
            HVX_Vector ihi = Q6_V_hi_W(iw[t]);
            HVX_Vector plo = Q6_Vh_vmpyi_VhVh(ilo, wlo_arr[t]);
            HVX_Vector phi = Q6_Vh_vmpyi_VhVh(ihi, whi_arr[t]);
            HVX_VectorPair wplo = Q6_Ww_vunpack_Vh(plo);
            HVX_VectorPair wphi = Q6_Ww_vunpack_Vh(phi);
            acc_w0 = Q6_Vw_vadd_VwVw(acc_w0, Q6_V_lo_W(wplo));
            acc_w1 = Q6_Vw_vadd_VwVw(acc_w1, Q6_V_hi_W(wplo));
            acc_w2 = Q6_Vw_vadd_VwVw(acc_w2, Q6_V_lo_W(wphi));
            acc_w3 = Q6_Vw_vadd_VwVw(acc_w3, Q6_V_hi_W(wphi));
        }

        /* + bias, ReLU clamp (int32), all still HVX vector ops */
        acc_w0 = Q6_Vw_vadd_VwVw(acc_w0, vbias0);
        acc_w1 = Q6_Vw_vadd_VwVw(acc_w1, vbias1);
        acc_w2 = Q6_Vw_vadd_VwVw(acc_w2, vbias2);
        acc_w3 = Q6_Vw_vadd_VwVw(acc_w3, vbias3);
        HVX_Vector vzero = Q6_V_vzero();
        acc_w0 = Q6_Vw_vmax_VwVw(acc_w0, vzero);
        acc_w1 = Q6_Vw_vmax_VwVw(acc_w1, vzero);
        acc_w2 = Q6_Vw_vmax_VwVw(acc_w2, vzero);
        acc_w3 = Q6_Vw_vmax_VwVw(acc_w3, vzero);

        *(HVX_Vector *)(accbuf +  0) = acc_w0;
        *(HVX_Vector *)(accbuf + 32) = acc_w1;
        *(HVX_Vector *)(accbuf + 64) = acc_w2;
        *(HVX_Vector *)(accbuf + 96) = acc_w3;

        int8_t *out_row = out + y * PW;
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C; c++) {
                int64_t v = (int64_t)accbuf[x * C + c] * (int64_t)mult;
                int64_t r;
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
