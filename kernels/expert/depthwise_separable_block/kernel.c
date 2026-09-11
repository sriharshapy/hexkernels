/*
 * depthwise_separable_block HVX kernel — expert
 *
 * H=8, W=8, C_in=8, C_out=8. Full MobileNet DSC block (BN-folded):
 * Stage 1: depthwise 3x3 SAME conv + bias + ReLU(int32) + requant -> int8
 *          (in-kernel scratch, no harness-visible mid buffer for this task).
 * Stage 2: pointwise 1x1 (C_in->C_out) + bias + ReLU(int32) + requant -> out.
 *
 * Stage 1: proven depthwise-NHWC idiom (see i8_depthwise_conv2d_bias_relu):
 * vectorize across (x*C+c) row-lane index, per-tap weight-broadcast vectors,
 * widen bytes via Q6_Wh_vunpack_Vb (sequential lanes), elementwise multiply
 * Q6_Vh_vmpyi_VhVh (NO cross-channel reduction), widen to int32 via
 * Q6_Ww_vunpack_Vh, sum 9 taps, add per-channel bias, ReLU-clamp
 * (Q6_Vw_vmax_VwVw vs zero), requantize to int8.
 *
 * Stage 2: real cross-channel matmul (1x1 conv) via vrmpy
 * (Q6_Vw_vrmpyacc_VwVbVb), reducing C_in in groups of 4, output vectorized
 * across C_out lanes per pixel; then add per-channel bias, ReLU-clamp,
 * requantize.
 */

#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in,
                      const int8_t *dw_wt, const int32_t *dw_bias,
                      const int8_t *pw_wt, const int32_t *pw_bias,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t dw_mult, int dw_shift, int8_t dw_zp,
                      int32_t pw_mult, int pw_shift, int8_t pw_zp) {
    const int PWrow = W * C_in;   /* 64 bytes/row for W=8,C_in=8 */
    const int ROW = 96;           /* 16B pad + PWrow + slack */

    int8_t mid[64 * 8] __attribute__((aligned(128))); /* [H][W][C_in], max 8*8*8=512 */

    /* ---------------- Stage 1: depthwise + bias + relu + requant -> mid --- */
    {
        int8_t padded[10 * 96] __attribute__((aligned(128)));
        memset(padded, 0, (size_t)(H + 2) * ROW);
        for (int y = 0; y < H; y++)
            memcpy(padded + (y + 1) * ROW + C_in, in + y * PWrow, (size_t)PWrow);

        int8_t wt_vecs[9 * 128] __attribute__((aligned(128)));
        for (int t = 0; t < 9; t++) {
            int64_t *dst64 = (int64_t *)(wt_vecs + t * 128);
            int64_t wp = 0;
            for (int c = 0; c < C_in; c++)
                wp |= ((int64_t)(uint8_t)dw_wt[c * 9 + t]) << (8 * c);
            for (int i = 0; i < 16; i++) dst64[i] = wp;
        }

        int32_t bias_buf[32] __attribute__((aligned(128)));
        for (int i = 0; i < 32; i++) bias_buf[i] = dw_bias[i % C_in];
        HVX_Vector vbias = *(const HVX_Vector *)(bias_buf);

        const int64_t half = (dw_shift > 0) ? ((int64_t)1 << (dw_shift - 1)) : 0;
        const int32_t izp = (int32_t)dw_zp;

        HVX_Vector wtap_arr[9];
        for (int t = 0; t < 9; t++)
            wtap_arr[t] = *(const HVX_Vector *)(wt_vecs + t * 128);

        HVX_Vector wlo_arr[9], whi_arr[9];
        for (int t = 0; t < 9; t++) {
            HVX_VectorPair wtw = Q6_Wh_vunpack_Vb(wtap_arr[t]);
            wlo_arr[t] = Q6_V_lo_W(wtw);
            whi_arr[t] = Q6_V_hi_W(wtw);
        }

        HVX_Vector vzero = Q6_V_vzero();
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

            acc_w0 = Q6_Vw_vadd_VwVw(acc_w0, vbias);
            acc_w1 = Q6_Vw_vadd_VwVw(acc_w1, vbias);
            acc_w2 = Q6_Vw_vadd_VwVw(acc_w2, vbias);
            acc_w3 = Q6_Vw_vadd_VwVw(acc_w3, vbias);
            acc_w0 = Q6_Vw_vmax_VwVw(acc_w0, vzero);
            acc_w1 = Q6_Vw_vmax_VwVw(acc_w1, vzero);
            acc_w2 = Q6_Vw_vmax_VwVw(acc_w2, vzero);
            acc_w3 = Q6_Vw_vmax_VwVw(acc_w3, vzero);

            *(HVX_Vector *)(accbuf +  0) = acc_w0;
            *(HVX_Vector *)(accbuf + 32) = acc_w1;
            *(HVX_Vector *)(accbuf + 64) = acc_w2;
            *(HVX_Vector *)(accbuf + 96) = acc_w3;

            int8_t *mid_row = mid + y * PWrow;
            for (int x = 0; x < W; x++) {
                for (int c = 0; c < C_in; c++) {
                    int64_t v = (int64_t)accbuf[x * C_in + c] * (int64_t)dw_mult;
                    int64_t r;
                    if (v >= 0) r = (v + half) >> dw_shift;
                    else        r = -((-v + half) >> dw_shift);
                    r += izp;
                    if (r >  127) r =  127;
                    if (r < -128) r = -128;
                    mid_row[x * C_in + c] = (int8_t)r;
                }
            }
        }
    }

    /* ---------------- Stage 2: pointwise + bias + relu + requant -> out --- */
    {
        int ngroups = (C_in + 3) / 4;

        int8_t pw_wtT[8 * 128] __attribute__((aligned(128)));
        memset(pw_wtT, 0, sizeof(pw_wtT));
        for (int g = 0; g < ngroups; g++) {
            int8_t *dst = pw_wtT + g * 128;
            for (int co = 0; co < C_out; co++) {
                for (int r = 0; r < 4; r++) {
                    int ci = g * 4 + r;
                    dst[co * 4 + r] = (ci < C_in) ? pw_wt[co * C_in + ci] : 0;
                }
            }
        }
        HVX_Vector vwT[8];
        for (int g = 0; g < ngroups; g++)
            vwT[g] = *(const HVX_Vector *)(pw_wtT + g * 128);

        int32_t bias_buf[32] __attribute__((aligned(128)));
        memset(bias_buf, 0, sizeof(bias_buf));
        for (int co = 0; co < C_out; co++) bias_buf[co] = pw_bias[co];
        HVX_Vector vbias = *(const HVX_Vector *)(bias_buf);
        HVX_Vector vzero = Q6_V_vzero();

        const int64_t half = (pw_shift > 0) ? ((int64_t)1 << (pw_shift - 1)) : 0;
        const int32_t izp = (int32_t)pw_zp;

        int npix = H * W;
        int32_t accbuf[32] __attribute__((aligned(128)));

        for (int p = 0; p < npix; p++) {
            const int8_t *midp = mid + p * C_in;
            int8_t *outp = out + p * C_out;

            HVX_Vector vacc = Q6_V_vzero();
            for (int g = 0; g < ngroups; g++) {
                uint32_t a0 = (uint8_t)midp[g*4 + 0];
                uint32_t a1 = (uint8_t)midp[g*4 + 1];
                uint32_t a2 = (uint8_t)midp[g*4 + 2];
                uint32_t a3 = (uint8_t)midp[g*4 + 3];
                uint32_t w = a0 | (a1 << 8) | (a2 << 16) | (a3 << 24);
                HVX_Vector va = Q6_V_vsplat_R(w);
                vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, va, vwT[g]);
            }
            vacc = Q6_Vw_vadd_VwVw(vacc, vbias);
            vacc = Q6_Vw_vmax_VwVw(vacc, vzero);
            *(HVX_Vector *)accbuf = vacc;

            for (int co = 0; co < C_out; co++) {
                int64_t v = (int64_t)accbuf[co] * (int64_t)pw_mult;
                int64_t r;
                if (v >= 0) r = (v + half) >> pw_shift;
                else        r = -((-v + half) >> pw_shift);
                r += izp;
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                outp[co] = (int8_t)r;
            }
        }
    }
}
