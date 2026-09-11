/* Near-miss: correct conv + relu but ignores the bias.
   Output is systematically wrong for any channel where bias != 0. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, const int32_t *bias,
                      int8_t *out,
                      int L, int K, int C) {
    (void)bias;  /* intentionally ignored */
    for (int i = 0; i < L; i++) {
        for (int c = 0; c < C; c++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)x[(i + k) * C + c] * (int32_t)taps[k * C + c];
            /* BUG: no bias added */
            int32_t relu_v = acc < 0 ? 0 : acc;
            if (relu_v >  127) relu_v =  127;
            if (relu_v < -128) relu_v = -128;
            out[i * C + c] = (int8_t)relu_v;
        }
    }
}
