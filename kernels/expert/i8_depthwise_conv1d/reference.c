#include <stdint.h>
/* Plainly correct scalar ground truth: depthwise 1D FIR (correlation).
   Each channel ch uses its own taps[ch*ntaps .. ch*ntaps+ntaps-1]. */
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int L, int C, int ntaps) {
    int xstride = L + ntaps - 1;
    for (int ch = 0; ch < C; ch++) {
        const int8_t *xch   = x    + ch * xstride;
        const int8_t *tapch = taps + ch * ntaps;
        int32_t      *och   = out  + ch * L;
        for (int i = 0; i < L; i++) {
            int32_t acc = 0;
            for (int j = 0; j < ntaps; j++)
                acc += (int32_t)xch[i + j] * (int32_t)tapch[j];
            och[i] = acc;
        }
    }
}
