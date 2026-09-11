#include <stdint.h>
/* Plainly correct scalar ground truth: strided 1D FIR (correlation).
   out[i] = sum_{j=0}^{ntaps-1} x[i*stride + j] * taps[j] */
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int n, int ntaps, int stride) {
    for (int i = 0; i < n; i++) {
        int32_t acc = 0;
        for (int j = 0; j < ntaps; j++)
            acc += (int32_t)x[i * stride + j] * (int32_t)taps[j];
        out[i] = acc;
    }
}
