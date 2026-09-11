#include <stdint.h>
/* Plainly correct scalar ground truth: 1D FIR, SAME padding (centered), correlation.
   pad_left = (ntaps-1)/2. out[i] = sum_j x[i - pad_left + j] * taps[j],
   treating out-of-bounds x as 0. */
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int n, int ntaps) {
    int pad_left = (ntaps - 1) / 2;
    for (int i = 0; i < n; i++) {
        int32_t acc = 0;
        for (int j = 0; j < ntaps; j++) {
            int xi = i - pad_left + j;
            if (xi >= 0 && xi < n)
                acc += (int32_t)x[xi] * (int32_t)taps[j];
            /* else: zero-padded, contributes 0 */
        }
        out[i] = acc;
    }
}
