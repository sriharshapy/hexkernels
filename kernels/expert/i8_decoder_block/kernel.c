/* EXPERT -- HVX-vectorized causal decoder block. Same fixed-point pipeline as
 * baseline.c, but every reduction (X.X^T, causal A.V, FFN up/down-proj) is one
 * HVX vrmpy dot instead of a scalar inner loop. Layouts transposed once up front
 * (Xt, W1t, W2t) so every dot is a contiguous <=128-byte vector load. */
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
/* unsigned-activation x signed-weight dot, nb<=128 bytes (128 = unmasked full vector). */
static inline int32_t vdot_us(const uint8_t *a, const int8_t *w, int nb) {
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
/* signed-activation x signed-weight dot, nb<=128 bytes. */
static inline int32_t vdot_ss(const int8_t *a, const int8_t *w, int nb) {
    HVX_Vector zero = Q6_V_vzero();
    HVX_Vector va = *(const HVX_UVector *)a;
    HVX_Vector vw = *(const HVX_UVector *)w;
    if (nb < 128) {
        HVX_VectorPred p = Q6_Q_vsetq_R(nb);
        va = Q6_V_vmux_QVV(p, va, zero);
        vw = Q6_V_vmux_QVV(p, vw, zero);
    }
    return hreduce32(Q6_Vw_vrmpy_VbVb(vw, va));
}

void candidate_kernel(const uint8_t *X,
                      const int8_t *W1, const int32_t *b1,
                      const int8_t *act_lut,
                      const int8_t *W2, const int32_t *b2,
                      const uint8_t *exp_lut,
                      int8_t *out, int S, int D, int Dff) {
    static int8_t  Xw [32*64]   HVX_ALIGN;   /* X reinterpreted as int8 (values 0..3) */
    static int8_t  Xt [64*32]   HVX_ALIGN;   /* X transposed [D x S] */
    static int8_t  W1t[128*64]  HVX_ALIGN;   /* W1 transposed [Dff x D] */
    static int8_t  W2t[64*128]  HVX_ALIGN;   /* W2 transposed [D x Dff] */
    static int8_t  scaled[32*32] HVX_ALIGN;
    static uint8_t probs[32*32]  HVX_ALIGN;
    static uint8_t h[32*64]      HVX_ALIGN;
    static int8_t  Hact[32*128]  HVX_ALIGN;

    for (int i = 0; i < S*D; i++) Xw[i] = (int8_t)X[i];
    for (int i = 0; i < S; i++) for (int d = 0; d < D; d++) Xt[d*S + i] = (int8_t)X[i*D + d];
    for (int k = 0; k < D;   k++) for (int j = 0; j < Dff; j++) W1t[j*D + k]   = W1[k*Dff + j];
    for (int k = 0; k < Dff; k++) for (int j = 0; j < D;   j++) W2t[j*Dff + k] = W2[k*D + j];

    for (int i = 0; i < S; i++) {
        for (int j = 0; j <= i; j++) {
            int raw = (int)vdot_us(X + i*D, Xw + j*D, D);
            scaled[i*S+j] = (int8_t)dec_clamp_i8(raw >> ATTN_SCALE_SHIFT);
        }
        int m = scaled[i*S+0];
        for (int j = 1; j <= i; j++) if (scaled[i*S+j] > m) m = scaled[i*S+j];
        int Sr = 0;
        for (int j = 0; j <= i; j++) {
            int diff = (int)scaled[i*S+j] - m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            probs[i*S+j] = e;
            Sr += (int)e;
        }
        int half = Sr / 2;
        for (int j = 0; j <= i; j++)
            probs[i*S+j] = (uint8_t)(((int)probs[i*S+j] * 255 + half) / Sr);
        for (int d = 0; d < D; d++) {
            /* causal window: only probs[i,0..i] and Xt[d,0..i] contribute. */
            int av = (int)vdot_us(probs + i*S, Xt + d*S, i + 1);
            int a = dec_clamp_i8(av >> ATTN_OUT_SHIFT);
            h[i*D+d] = (uint8_t)dec_clamp_u8((int)X[i*D+d] + a);
        }
    }
    for (int i = 0; i < S; i++)
        for (int j = 0; j < Dff; j++) {
            int acc1 = (int)vdot_us(h + i*D, W1t + j*D, D);
            int p1 = (acc1 >> FFN_SH1) + b1[j];
            int idx = dec_clamp_i8(p1) + 128;
            Hact[i*Dff+j] = act_lut[idx];
        }
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++) {
            int acc2 = (int)vdot_ss(Hact + i*Dff, W2t + j*Dff, Dff);
            int p2 = (acc2 >> FFN_SH2) + b2[j];
            int fv = dec_sat_i8(p2);
            out[i*D+j] = dec_sat_i8((int)h[i*D+j] + fv);
        }
}
