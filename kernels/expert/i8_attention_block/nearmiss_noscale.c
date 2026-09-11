/* NEAR-MISS (must score INCORRECT): omits the attention SCALE. It softmaxes the
 * raw requant scores directly (no >>4 shift), so scores saturate to +-127 and the
 * softmax collapses toward near-one-hot -- different probs, different output.
 * Otherwise identical to the HVX baseline.
 *
 * DENOMINATOR baseline -- the WHOLE attention block in HVX, NO HMX matrix engine.
 *   QK^T : row-row dot products via Q6_Vw_vrmpyacc_VwVbVb (int8 4-deep reduce)
 *   scale+softmax : fixed-point LUT, row-wise over the key axis (scalar reduce --
 *                   softmax's max/sum are inherently sequential per row)
 *   A.V  : transpose V once, then probs(uint8) . V^T row-row dots via
 *          Q6_Vw_vrmpyacc_VwVubVb (unsigned activation x signed weight)
 * The same 0x40-config requant as the reference is applied after each matmul.
 * This is the honest single-engine baseline the HMX-composed expert must beat. */
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
static inline int sx12(int field) {
    int v = field & 0xFFF; if (v & 0x800) v -= 0x1000; return v;
}
static inline int clamp_i8(int v) {
    if (v > 127) return 127; if (v < -128) return -128; return v;
}

void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                      uint16_t *out, int S, int D, const uint8_t *exp_lut) {
    static int8_t  scaled[64*64] HVX_ALIGN;
    static uint8_t probs[64*64]  HVX_ALIGN;
    static int8_t  Vt[64*64]     HVX_ALIGN;   /* V transposed: Vt[d][j] = V[j][d] */

    const HVX_VectorPred qn = Q6_Q_vsetq_R(D);   /* first D bytes valid */
    const HVX_VectorPred kn = Q6_Q_vsetq_R(S);   /* first S bytes valid (A.V reduce) */
    const HVX_Vector zero = Q6_V_vzero();

    /* Stage 1: scores = Q.K^T, requant, scale + clamp -> int8 scaled */
    for (int i = 0; i < S; i++) {
        HVX_Vector vq = Q6_V_vmux_QVV(qn, *(const HVX_UVector *)(Q + i*D), zero);
        for (int j = 0; j < S; j++) {
            HVX_Vector vk = Q6_V_vmux_QVV(qn, *(const HVX_UVector *)(K + j*D), zero);
            int32_t acc = hreduce32(Q6_Vw_vrmpyacc_VwVbVb(zero, vq, vk));
            int s12 = hvx_hmx_requant_0x40(acc);
            scaled[i*S+j] = (int8_t)clamp_i8(sx12(s12));  /* BUG: no >>4 scale */
        }
    }

    /* Stage 2: row-wise softmax over key axis (fixed-point LUT) */
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

    /* Stage 3: out = probs . V.  Transpose V (Vt[d][j]=V[j][d]) so each output
     * cell out[i][d] = probs[i][:] . Vt[d][:] is a contiguous row-row dot. */
    for (int j = 0; j < S; j++)
        for (int d = 0; d < D; d++)
            Vt[d*S + j] = V[j*D + d];

    for (int i = 0; i < S; i++) {
        HVX_Vector vp = Q6_V_vmux_QVV(kn, *(const HVX_UVector *)(probs + i*S), zero);
        for (int d = 0; d < D; d++) {
            HVX_Vector vv = Q6_V_vmux_QVV(kn, *(const HVX_UVector *)(Vt + d*S), zero);
            int32_t acc = hreduce32(Q6_Vw_vrmpyacc_VwVubVb(zero, vp, vv));
            out[i*D + d] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }
    }
}
