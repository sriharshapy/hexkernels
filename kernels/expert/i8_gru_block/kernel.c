/* EXPERT (achievability bar) -- HVX-vectorized MULTI-STEP integer GRU block.
 * Each gate row's dot is SPLIT into two masked vrmpy dots (the x-part of
 * length I, the h-part of length H) that are summed -- no 128-byte concat
 * buffer/memset per timestep (measured faster than the concat-vector
 * approach at these small dims: no per-step memset+copy overhead). The
 * fused z/r gate matmul walks a SINGLE weight tensor Wzr for 2H rows in one
 * loop, and the whole cell recurs T times, propagating h_prev across
 * timesteps. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "kernel_api.h"

static inline int32_t gru_requant(int32_t acc, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return r;
}
static inline int32_t gru_clamp8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return v;
}
static inline int32_t hreduce32(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}
/* signed-signed masked dot, nb<=127 bytes (never a full 128 here: I=H=16). */
static inline int32_t vdot_ss(const int8_t *a, const int8_t *w, int nb) {
    HVX_Vector zero = Q6_V_vzero();
    HVX_Vector va = *(const HVX_UVector *)a;
    HVX_Vector vw = *(const HVX_UVector *)w;
    HVX_VectorPred p = Q6_Q_vsetq_R(nb);
    va = Q6_V_vmux_QVV(p, va, zero);
    vw = Q6_V_vmux_QVV(p, vw, zero);
    return hreduce32(Q6_Vw_vrmpy_VbVb(vw, va));
}

void candidate_kernel(
    const int8_t *Wzr, const int8_t *Wn,
    const int32_t *bzr, const int32_t *bn,
    const int8_t *x, const int8_t *h0, int8_t *h_out,
    int T, int H, int I, int32_t mult, int shift, int8_t zp,
    const int8_t *sig_lut, const int8_t *tanh_lut) {
    int C = I + H;
    int8_t hprev[64];
    for (int j = 0; j < H; j++) hprev[j] = h0[j];

    for (int t = 0; t < T; t++) {
        const int8_t *xt = x + t*I;
        int8_t z[64], r[64], rh[64], n[64];

        for (int j = 0; j < 2*H; j++) {
            int32_t acc = bzr[j]
                        + (int32_t)vdot_ss(xt,    Wzr + j*C,     I)
                        + (int32_t)vdot_ss(hprev, Wzr + j*C + I, H);
            int8_t q = (int8_t)gru_requant(acc, mult, shift, zp);
            if (j < H) z[j]   = sig_lut[(uint8_t)q];
            else       r[j-H] = sig_lut[(uint8_t)q];
        }

        for (int j = 0; j < H; j++)
            rh[j] = (int8_t)gru_clamp8(((int32_t)(r[j] + 128) * (int32_t)hprev[j] + 64) >> 7);

        for (int j = 0; j < H; j++) {
            int32_t acc = bn[j]
                        + (int32_t)vdot_ss(xt, Wn + j*C,     I)
                        + (int32_t)vdot_ss(rh, Wn + j*C + I, H);
            n[j] = tanh_lut[(uint8_t)gru_requant(acc, mult, shift, zp)];
        }

        for (int j = 0; j < H; j++) {
            int32_t blend = ((int32_t)(128 - z[j]) * (int32_t)hprev[j]
                           + (int32_t)(z[j] + 128) * (int32_t)n[j] + 128) >> 8;
            int8_t ht = (int8_t)gru_clamp8(blend);
            h_out[t*H+j] = ht;
            hprev[j] = ht;
        }
    }
}
