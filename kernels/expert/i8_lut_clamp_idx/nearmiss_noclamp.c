/* Near-miss: no clamping — ignores lo/hi, uses raw (uint8_t)in[i] as index.
   Wrong for inputs outside [lo,hi]: produces lut[(uint8_t)in[i]] instead of
   lut[clamp((uint8_t)in[i], lo, hi)]. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n,
                      const int8_t *lut, uint8_t lo, uint8_t hi){
    (void)lo; (void)hi;
    for (int i = 0; i < n; i++) out[i] = lut[(uint8_t)in[i]];
}
