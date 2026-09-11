/* EXPERT -- HVX-vectorized MLP-Mixer block. Same fixed-point pipeline as
 * baseline.c, but every reduction is one HVX vrmpy dot instead of a scalar
 * inner loop. Layouts transposed once up front (Xt, TMht, Wc1t, Wc2t) so
 * every dot is a contiguous <=128-byte vector load, matching each stage's
 * actual reduction axis (S or St for token-mixing, D or Dff for channel-mixing). */
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
                      const int8_t *Wt1, const int32_t *bt1,
                      const int8_t *Wt2, const int32_t *bt2,
                      const int8_t *Wc1, const int32_t *bc1,
                      const int8_t *Wc2, const int32_t *bc2,
                      int8_t *out, int S, int D, int St, int Dff) {
    static int8_t  Xt  [64*32]  HVX_ALIGN;   /* [D x S] */
    static uint8_t TMh [64*64]  HVX_ALIGN;   /* [St x D] */
    static uint8_t TMht[64*64]  HVX_ALIGN;   /* [D x St] */
    static int8_t  Y   [32*64]  HVX_ALIGN;   /* [S x D] */
    static int8_t  Wc1t[64*64]  HVX_ALIGN;   /* [Dff x D] */
    static int8_t  Wc2t[64*64]  HVX_ALIGN;   /* [D x Dff] */
    static uint8_t CMh [32*64]  HVX_ALIGN;   /* [S x Dff] */

    for (int s = 0; s < S; s++) for (int d = 0; d < D; d++) Xt[d*S + s] = (int8_t)X[s*D + d];
    for (int d = 0; d < D; d++) for (int j = 0; j < Dff; j++) Wc1t[j*D + d] = Wc1[d*Dff + j];
    for (int j = 0; j < Dff; j++) for (int d = 0; d < D; d++) Wc2t[d*Dff + j] = Wc2[j*D + d];

    /* ---- Token-mixing (mix over S) ---- */
    for (int d = 0; d < D; d++)
        for (int sp = 0; sp < St; sp++) {
            int acc = (int)vdot_us((const uint8_t *)(Xt + d*S), Wt1 + sp*S, S);
            /* Xt holds X values (0..3) reinterpreted as int8; safe as "unsigned"
             * activation operand since all stored values are non-negative. */
            int p = (acc >> TM_SH1) + bt1[sp];
            TMh[sp*D+d] = (uint8_t)mx_relu_i8(p);
        }
    for (int sp = 0; sp < St; sp++) for (int d = 0; d < D; d++) TMht[d*St + sp] = TMh[sp*D + d];
    for (int d = 0; d < D; d++)
        for (int s = 0; s < S; s++) {
            int acc2 = (int)vdot_us(TMht + d*St, Wt2 + s*St, St);
            int p2 = (acc2 >> TM_SH2) + bt2[s];
            int tmo = mx_sat_i8(p2);
            Y[s*D+d] = mx_sat_i8((int)X[s*D+d] + tmo);
        }

    /* ---- Channel-mixing (mix over D) ---- */
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < Dff; j++) {
            int acc = (int)vdot_ss(Y + s*D, Wc1t + j*D, D);
            int p = (acc >> CM_SH1) + bc1[j];
            CMh[s*Dff+j] = (uint8_t)mx_relu_i8(p);
        }
        for (int d = 0; d < D; d++) {
            int acc2 = (int)vdot_us(CMh + s*Dff, Wc2t + d*Dff, Dff);
            int p2 = (acc2 >> CM_SH2) + bc2[d];
            int cmo = mx_sat_i8(p2);
            out[s*D+d] = mx_sat_i8((int)Y[s*D+d] + cmo);
        }
    }
}
