/* Near-miss: hardcodes requant params (mult=5, shift=4, zp=0) instead of
   reading them from arguments. Passes the first sweep set but fails the second. */
#include <stdint.h>

static int8_t sat8_hc(int32_t x) {
    if (x >  127) return  127;
    if (x < -128) return -128;
    return (int8_t)x;
}

void candidate_kernel(const int8_t  *A,
                      const int8_t  *W1, const int32_t *b1,
                      const int8_t  *W2, const int32_t *b2,
                      int8_t        *out,
                      int M, int K, int V, int D,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp; /* BUG: ignore runtime requant params */
    int8_t hid[8*32];

    /* Layer 1: correct */
    for (int i = 0; i < M; i++) {
        for (int v = 0; v < V; v++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)W1[k*V+v];
            int64_t biased = (int64_t)acc + (int64_t)b1[v];
            if (biased < 0) biased = 0;
            hid[i*V+v] = sat8_hc((int32_t)biased);
        }
    }

    /* Layer 2: hardcoded mult=5, shift=4, zp=0 */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) {
            int32_t acc = 0;
            for (int v = 0; v < V; v++)
                acc += (int32_t)hid[i*V+v] * (int32_t)W2[v*D+j];
            int64_t biased = (int64_t)acc + (int64_t)b2[j];
            /* BUG: hardcoded mult=5, shift=4 */
            int64_t vv = biased * 5LL;
            int64_t half = 8LL; /* 1 << (4-1) */
            int64_t r = (vv >= 0) ? ((vv + half) >> 4) : -(((-vv) + half) >> 4);
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            out[i*D+j] = (int8_t)r;
        }
    }
}
