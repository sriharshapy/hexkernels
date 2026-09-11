/* NEAR-MISS (must score INCORRECT): DROPS the second residual add. A transformer
 * encoder block adds the FFN output back to the residual stream (out = h + ffn_out);
 * this version outputs only the FFN result (out = ffn_out), the classic "forgot the
 * post-FFN skip connection" bug. Different output. Otherwise the correct HVX baseline. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline int32_t hreduce32(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}
static inline int32_t vdot(const uint8_t *a, const int8_t *w, int nb) {
    HVX_Vector zero = Q6_V_vzero();
    HVX_Vector va = *(const HVX_UVector *)a;
    HVX_Vector vw = *(const HVX_UVector *)w;
    if (nb < 128) {
        HVX_VectorPred p = Q6_Q_vsetq_R(nb);
        va = Q6_V_vmux_QVV(p, va, zero);
        vw = Q6_V_vmux_QVV(p, vw, zero);
    }
    return hreduce32(Q6_Vw_vrmpyacc_VwVubVb(zero, va, vw));
}

void candidate_kernel(const uint8_t *X,
                      const int8_t *W1, const int32_t *b1,
                      const int8_t *W2, const int32_t *b2,
                      const uint8_t *exp_lut,
                      int8_t *out, int S, int D, int Dff) {
    static int8_t  Xw[64*64]     HVX_ALIGN;
    static int8_t  Xt[64*64]     HVX_ALIGN;
    static int8_t  W1t[128*64]   HVX_ALIGN;
    static int8_t  W2t[64*128]   HVX_ALIGN;
    static int8_t  scaled[64*64] HVX_ALIGN;
    static uint8_t probs[64*64]  HVX_ALIGN;
    static uint8_t h[64*64]      HVX_ALIGN;
    static uint8_t Hb[64*128]    HVX_ALIGN;

    for (int i = 0; i < S*D; i++) Xw[i] = (int8_t)X[i];
    for (int i = 0; i < S; i++) for (int d = 0; d < D; d++) Xt[d*S + i] = (int8_t)X[i*D + d];
    for (int k = 0; k < D;   k++) for (int j = 0; j < Dff; j++) W1t[j*D + k]   = W1[k*Dff + j];
    for (int k = 0; k < Dff; k++) for (int j = 0; j < D;   j++) W2t[j*Dff + k] = W2[k*D + j];

    /* Self-attention */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            int acc = (int)vdot(X + i*D, Xw + j*D, D);
            scaled[i*S+j] = (int8_t)tf_clamp_i8(tf_sx12(hvx_hmx_requant_0x40(acc)) >> ATTN_SCALE_SHIFT);
        }
    for (int i = 0; i < S; i++) {
        int m = scaled[i*S+0];
        for (int j = 1; j < S; j++) if (scaled[i*S+j] > m) m = scaled[i*S+j];
        int Sr = 0;
        for (int j = 0; j < S; j++) {
            int diff = (int)scaled[i*S+j] - m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            probs[i*S+j] = e;
            Sr += (int)e;
        }
        int half = Sr / 2;
        for (int j = 0; j < S; j++)
            probs[i*S+j] = (uint8_t)(((int)probs[i*S+j] * 255 + half) / Sr);
    }
    for (int i = 0; i < S; i++)
        for (int d = 0; d < D; d++) {
            int acc = (int)vdot(probs + i*S, Xt + d*S, S);
            int a = tf_clamp_i8(tf_sx12(hvx_hmx_requant_0x40(acc)) >> ATTN_OUT_SHIFT);
            h[i*D+d] = (uint8_t)tf_clamp_u8((int)X[i*D+d] + a);   /* residual 1 */
        }

    /* FFN */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < Dff; j++) {
            int acc1 = (int)vdot(h + i*D, W1t + j*D, D);
            int r1 = tf_sx12(hvx_hmx_requant_0x40(acc1));
            Hb[i*Dff+j] = (uint8_t)tf_relu_requant(r1 + b1[j]);
        }
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++) {
            int acc2 = (int)vdot(Hb + i*Dff, W2t + j*Dff, Dff);
            int r2 = tf_sx12(hvx_hmx_requant_0x40(acc2));
            int fv = (r2 + b2[j]) >> FFN_SH2;
            out[i*D+j] = tf_sat_i8(fv);           /* BUG: residual 2 dropped (no + h) */
        }
}
