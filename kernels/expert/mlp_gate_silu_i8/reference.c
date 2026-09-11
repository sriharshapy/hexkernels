#include <stdint.h>
void candidate_kernel(const int8_t *gate, const int8_t *up, const int8_t *silu_lut,
                      int8_t *out, int N, int32_t scale_mult, int scale_shift) {
    for (int i = 0; i < N; i++) {
        uint8_t idx = (uint8_t)((int)gate[i] + 128);
        int32_t g = silu_lut[idx];
        int32_t prod = g * (int32_t)up[i];
        int64_t r    = (int64_t)prod * (int64_t)scale_mult;
        int64_t half = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0;
        int64_t q    = (r >= 0) ? ((r + half) >> scale_shift)
                                 : -(((-r) + half) >> scale_shift);
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}
