#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *x,
                      const int32_t *bias, int8_t *out,
                      int M, int K,
                      int32_t mult, int shift, int8_t zp) {
    for (int i = 0; i < M; i++) {
        /* Step 1: dot product */
        int32_t acc = 0;
        for (int k = 0; k < K; k++)
            acc += (int32_t)A[i*K+k] * (int32_t)x[k];
        /* Step 2: add per-row bias */
        int64_t biased = (int64_t)acc + (int64_t)bias[i];
        /* Step 3: requantize */
        int64_t v    = biased * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
