/* Near-miss: correct conv + bias but skips ReLU.
   Wherever acc+bias < 0, output should be 0 but this returns the negative value. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, const int32_t *bias,
                      int8_t *out,
                      int L, int K, int C) {
    for (int i = 0; i < L; i++) {
        for (int c = 0; c < C; c++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)x[(i + k) * C + c] * (int32_t)taps[k * C + c];
            int32_t biased = acc + bias[c];
            /* BUG: no ReLU */
            if (biased >  127) biased =  127;
            if (biased < -128) biased = -128;
            out[i * C + c] = (int8_t)biased;
        }
    }
}
