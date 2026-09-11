/* Near-miss: correct FIR + bias + saturate but SKIPS the relu step.
   Wherever acc+bias < 0, produces negative output instead of 0.
   bias_buf[0] = -200000 forces many outputs to be negative for ch0. */
#include <stdint.h>
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
            /* BUG: no relu -- negative values pass through */
            if (biased >  127) biased =  127;
            if (biased < -128) biased = -128;
            och[i] = (int8_t)biased;
        }
    }
}
