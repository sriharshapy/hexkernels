/* Near-miss: correct FIR + relu + saturate but SKIPS adding bias.
   Wherever bias is non-zero the output diverges from reference.
   bias_buf[1] = 200000 guarantees ch1 outputs are wrong (all saturate to 127
   with bias, but un-biased FIR can be much smaller). */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, const int32_t *bias,
                      int8_t *out, int L, int C, int ntaps) {
    (void)bias;  /* BUG: ignores bias entirely */
    int xstride = L + ntaps - 1;
    for (int ch = 0; ch < C; ch++) {
        const int8_t *xch   = x    + ch * xstride;
        const int8_t *tapch = taps + ch * ntaps;
        int8_t       *och   = out  + ch * L;
        for (int i = 0; i < L; i++) {
            int32_t acc = 0;
            for (int j = 0; j < ntaps; j++)
                acc += (int32_t)xch[i + j] * (int32_t)tapch[j];
            if (acc < 0) acc = 0; /* relu without bias */
            if (acc >  127) acc =  127;
            och[i] = (int8_t)acc;
        }
    }
}
