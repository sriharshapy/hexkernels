#include <stdint.h>
/* Scalar ground truth: depthwise 1D FIR (correlation) + bias + relu + saturate -> int8. */
void candidate_kernel(const int8_t *x, const int8_t *taps, const int32_t *bias,
                      int8_t *out, int L, int C, int ntaps) {
    int xstride = L + ntaps - 1;
    for (int ch = 0; ch < C; ch++) {
        const int8_t *xch   = x    + ch * xstride;
        const int8_t *tapch = taps + ch * ntaps;
        int8_t       *och   = out  + ch * L;
        int32_t       b     = bias[ch];
        for (int i = 0; i < L; i++) {
            int32_t acc = 0;
            for (int j = 0; j < ntaps; j++)
                acc += (int32_t)xch[i + j] * (int32_t)tapch[j];
            int32_t biased = acc + b;
            /* relu: clamp negatives to zero */
            if (biased < 0) biased = 0;
            /* saturate int32 to int8 */
            if (biased >  127) biased =  127;
            och[i] = (int8_t)biased;
        }
    }
}
