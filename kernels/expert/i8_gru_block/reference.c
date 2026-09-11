/*
 * DENOMINATOR baseline: plain scalar multi-step integer GRU block with fused
 * update/reset gate matmul. See kernel_api.h for the complete bit-exact
 * per-timestep equation.
 */
#include "kernel_api.h"

static int8_t gru_requant(int32_t acc, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}
static int8_t gru_clamp8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

void candidate_kernel(
    const int8_t  *Wzr, const int8_t *Wn,
    const int32_t *bzr, const int32_t *bn,
    const int8_t  *x,   const int8_t  *h0,
    int8_t        *h_out,
    int T, int H, int I,
    int32_t mult, int shift, int8_t zp,
    const int8_t  *sig_lut, const int8_t *tanh_lut
) {
    int C = I + H;
    int8_t hprev[64];
    for (int j = 0; j < H; j++) hprev[j] = h0[j];

    for (int t = 0; t < T; t++) {
        const int8_t *xt = x + t*I;
        int8_t z[64], r[64], rh[64], n[64];

        for (int j = 0; j < 2*H; j++) {
            int32_t acc = bzr[j];
            for (int k = 0; k < I; k++) acc += (int32_t)Wzr[j*C+k]   * (int32_t)xt[k];
            for (int k = 0; k < H; k++) acc += (int32_t)Wzr[j*C+I+k] * (int32_t)hprev[k];
            int8_t q = gru_requant(acc, mult, shift, zp);
            if (j < H) z[j]   = sig_lut[(uint8_t)q];
            else       r[j-H] = sig_lut[(uint8_t)q];
        }

        for (int j = 0; j < H; j++) {
            int32_t blend = ((int32_t)(r[j] + 128) * (int32_t)hprev[j] + 64) >> 7;
            rh[j] = gru_clamp8(blend);
        }

        for (int j = 0; j < H; j++) {
            int32_t acc = bn[j];
            for (int k = 0; k < I; k++) acc += (int32_t)Wn[j*C+k]   * (int32_t)xt[k];
            for (int k = 0; k < H; k++) acc += (int32_t)Wn[j*C+I+k] * (int32_t)rh[k];
            n[j] = tanh_lut[(uint8_t)gru_requant(acc, mult, shift, zp)];
        }

        for (int j = 0; j < H; j++) {
            int32_t blend = ((int32_t)(128 - z[j]) * (int32_t)hprev[j]
                           + (int32_t)(z[j] + 128) * (int32_t)n[j] + 128) >> 8;
            int8_t ht = gru_clamp8(blend);
            h_out[t*H+j] = ht;
            hprev[j] = ht;
        }
    }
}
