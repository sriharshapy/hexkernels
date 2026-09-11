#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int C, int L, int K,
                      int32_t mult, int shift, int8_t zp) {
    for (int c = 0; c < C; c++) {
        const int8_t *xc   = x    + c * (L + K - 1);
        const int8_t *tapc = taps + c * K;
        int8_t       *oc   = out  + c * L;
        for (int i = 0; i < L; i++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)xc[i + k] * (int32_t)tapc[k];
            /* requantize: round-half-away-from-zero */
            int64_t v    = (int64_t)acc * (int64_t)mult;
            int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
            int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
            r += zp;
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            oc[i] = (int8_t)r;
        }
    }
}
