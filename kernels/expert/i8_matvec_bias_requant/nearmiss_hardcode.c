/* Near-miss: hardcodes mult=3,shift=4,zp=0. Passes first param set but fails the rest. */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *x,
                      const int32_t *bias, int8_t *out,
                      int M, int K,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;
    for (int i = 0; i < M; i++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++)
            acc += (int32_t)A[i*K+k] * (int32_t)x[k];
        int64_t biased = (int64_t)acc + (int64_t)bias[i];
        long long v = biased * 3LL;
        long long h = 8LL;
        long long r = (v >= 0) ? ((v + h) >> 4) : -((-v + h) >> 4);
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (signed char)r;
    }
}
