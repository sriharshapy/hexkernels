/* Near-miss: correct depthwise conv but skips requantization.
   acc is truncated directly to int8 (saturate), ignoring mult/shift/zp.
   Will fail whenever requant changes the value (almost always). */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int C, int L, int K,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;
    for (int c = 0; c < C; c++) {
        const int8_t *xc   = x    + c * (L + K - 1);
        const int8_t *tapc = taps + c * K;
        int8_t       *oc   = out  + c * L;
        for (int i = 0; i < L; i++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)xc[i + k] * (int32_t)tapc[k];
            /* BUG: no requant, just saturate raw acc */
            if (acc >  127) acc =  127;
            if (acc < -128) acc = -128;
            oc[i] = (int8_t)acc;
        }
    }
}
