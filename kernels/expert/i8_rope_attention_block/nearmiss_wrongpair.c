/* NEAR-MISS (must score INCORRECT): applies the RoPE rotation to the WRONG feature
 * axis. Correct RoPE rotates the pair (d, d+HALF) -- the two halves of the head
 * dimension. This version rotates ADJACENT pairs (2d, 2d+1) instead, a classic
 * "wrong rotate_half layout" bug. The rotation is applied (tables are read), but the
 * paired dimensions are wrong, so Qr/Kr and the whole output differ. Otherwise the
 * correct HVX baseline. */
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
static inline int clamp_u8(int v) {
    if (v > 255) return 255; if (v < 0) return 0; return v;
}

void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                      const int8_t *cos_lut, const int8_t *sin_lut,
                      uint16_t *out, int S, int D, int rot_shift,
                      const uint8_t *exp_lut) {
    static uint8_t Qr[64*64]     HVX_ALIGN;
    static int8_t  Kr[64*64]     HVX_ALIGN;
    static int8_t  scaled[64*64] HVX_ALIGN;
    static uint8_t probs[64*64]  HVX_ALIGN;
    static int8_t  Vt[64*64]     HVX_ALIGN;

    const int HALF = D / 2;
    const HVX_VectorPred qn = Q6_Q_vsetq_R(D);
    const HVX_VectorPred kn = Q6_Q_vsetq_R(S);
    const HVX_Vector zero = Q6_V_vzero();

    /* BUG: rotate ADJACENT pairs (2d, 2d+1) instead of (d, d+HALF). */
    for (int i = 0; i < S; i++) {
        const int8_t *ci = cos_lut + i*HALF;
        const int8_t *si = sin_lut + i*HALF;
        for (int d = 0; d < HALF; d++) {
            int c = ci[d], s = si[d];
            int q0 = Q[i*D + 2*d], q1 = Q[i*D + 2*d + 1];
            Qr[i*D + 2*d]     = (uint8_t)clamp_u8((q0*c - q1*s) >> rot_shift);
            Qr[i*D + 2*d + 1] = (uint8_t)clamp_u8((q0*s + q1*c) >> rot_shift);
            int k0 = K[i*D + 2*d], k1 = K[i*D + 2*d + 1];
            Kr[i*D + 2*d]     = (int8_t)clamp_i8((k0*c - k1*s) >> rot_shift);
            Kr[i*D + 2*d + 1] = (int8_t)clamp_i8((k0*s + k1*c) >> rot_shift);
        }
    }

    for (int i = 0; i < S; i++) {
        HVX_Vector vq = Q6_V_vmux_QVV(qn, *(const HVX_UVector *)(Qr + i*D), zero);
        for (int j = 0; j < S; j++) {
            HVX_Vector vk = Q6_V_vmux_QVV(qn, *(const HVX_UVector *)(Kr + j*D), zero);
            int32_t acc = hreduce32(Q6_Vw_vrmpyacc_VwVubVb(zero, vq, vk));
            int s12 = hvx_hmx_requant_0x40(acc);
            scaled[i*S+j] = (int8_t)clamp_i8(sx12(s12) >> 4);
        }
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
