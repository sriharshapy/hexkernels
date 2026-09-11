/* Near-miss: no clamping — accesses lut[in[i]] directly.
   Produces OOB reads / wrong output for negative or too-large indices. */
#include <stdint.h>
void candidate_kernel(const int16_t *in, int8_t *out, int n,
                      const int8_t *lut, int lutsize){
    (void)lutsize;
    for (int i = 0; i < n; i++) out[i] = lut[(int)in[i]];
}
