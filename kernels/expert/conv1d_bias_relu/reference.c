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
            /* ReLU */
            int32_t relu_v = biased < 0 ? 0 : biased;
            /* Saturate to int8 */
            if (relu_v >  127) relu_v =  127;
            if (relu_v < -128) relu_v = -128;
            out[i * C + c] = (int8_t)relu_v;
        }
    }
}
