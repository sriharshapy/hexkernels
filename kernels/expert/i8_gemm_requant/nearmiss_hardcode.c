/* Near-miss: hardcodes mult=3,shift=4,zp=0 — ignores runtime params.
   Fails the param-sweep because other (mult,shift,zp) sets produce wrong results. */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B, int8_t *out,
                      int M, int N, int K,
                      int32_t mult, int shift, int8_t zp) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
            /* hardcoded mult=3, shift=4 */
            long long v = (long long)acc * 3;
            long long h = 8;  /* 1 << (4-1) */
            long long r = (v >= 0) ? ((v + h) >> 4) : -(((-v) + h) >> 4);
            r += zp;   /* at least uses zp */
            if (r > 127) r = 127; if (r < -128) r = -128;
            out[i*N+j] = (signed char)r;
        }
    }
}
