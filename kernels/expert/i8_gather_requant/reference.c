#include <stdint.h>
/*
 * i8_gather_requant baseline (scalar reference implementation).
 * out[i] = saturate_i8( round( (int32_t)table[idx[i]] * mult / 2^shift ) + zp )
 * Rounding: round-half-away-from-zero.
 */
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    for (int i = 0; i < n; i++) {
        int32_t raw  = (int32_t)table[idx[i]];
        int64_t v    = (int64_t)raw * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift)
                                 : -(((-v) + half) >> shift);
        r += (int64_t)zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
