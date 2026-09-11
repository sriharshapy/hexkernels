/* Near-miss: swaps update gate (z) and reset gate (r) roles.
   Uses r for the output blend and z for gating the recurrent input.
   Produces wrong h_t whenever z != r (virtually always). */
#include <stdint.h>

void candidate_kernel(
    const int8_t  *Wz,
    const int8_t  *Wr,
    const int8_t  *Wn,
    const int32_t *bz,
    const int32_t *br,
    const int32_t *bn,
    const int8_t  *x,
    const int8_t  *h_prev,
    int8_t        *h_t,
    int            H,
    int            I,
    int32_t        mult,
    int            shift,
    int8_t         zp,
    const int8_t  *sig_lut,
    const int8_t  *tanh_lut
) {
    int C = I + H;
    int8_t z[64], r[64], rh[64], n[64];

#define REQUANT(acc_) do { \
    int64_t v_    = (int64_t)(acc_) * (int64_t)mult; \
    int64_t half_ = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0; \
    int64_t r_    = (v_ >= 0) ? ((v_ + half_) >> shift) \
                               : -(((-v_) + half_) >> shift); \
    r_ += zp; \
    if (r_ >  127) r_ =  127; \
    if (r_ < -128) r_ = -128; \
    pre_ = (int8_t)r_; \
} while(0)

    for (int j = 0; j < H; j++) {
        int32_t acc = bz[j];
        for (int k = 0; k < I; k++) acc += (int32_t)Wz[j*C+k]   * (int32_t)x[k];
        for (int k = 0; k < H; k++) acc += (int32_t)Wz[j*C+I+k] * (int32_t)h_prev[k];
        int8_t pre_; REQUANT(acc);
        z[j] = sig_lut[(uint8_t)pre_];
    }
    for (int j = 0; j < H; j++) {
        int32_t acc = br[j];
        for (int k = 0; k < I; k++) acc += (int32_t)Wr[j*C+k]   * (int32_t)x[k];
        for (int k = 0; k < H; k++) acc += (int32_t)Wr[j*C+I+k] * (int32_t)h_prev[k];
        int8_t pre_; REQUANT(acc);
        r[j] = sig_lut[(uint8_t)pre_];
    }

    /* BUG: use z (update gate) to gate h_prev instead of r (reset gate) */
    for (int j = 0; j < H; j++) {
        int32_t blend = ((int32_t)(z[j] + 128) * (int32_t)h_prev[j] + 64) >> 7;
        if (blend >  127) blend =  127;
        if (blend < -128) blend = -128;
        rh[j] = (int8_t)blend;
    }

    for (int j = 0; j < H; j++) {
        int32_t acc = bn[j];
        for (int k = 0; k < I; k++) acc += (int32_t)Wn[j*C+k]   * (int32_t)x[k];
        for (int k = 0; k < H; k++) acc += (int32_t)Wn[j*C+I+k] * (int32_t)rh[k];
        int8_t pre_; REQUANT(acc);
        n[j] = tanh_lut[(uint8_t)pre_];
    }

    /* BUG: use r (reset gate) for the output blend instead of z (update gate) */
    for (int j = 0; j < H; j++) {
        int32_t blend = ((int32_t)(128 - r[j]) * (int32_t)h_prev[j]
                       + (int32_t)(r[j] + 128) * (int32_t)n[j] + 128) >> 8;
        if (blend >  127) blend =  127;
        if (blend < -128) blend = -128;
        h_t[j] = (int8_t)blend;
    }

#undef REQUANT
}
