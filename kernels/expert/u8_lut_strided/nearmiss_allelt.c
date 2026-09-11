/* Near-miss: applies LUT to ALL elements instead of only every k-th.
   Wrong for elements where i%k != 0 (those should be copied unchanged). */
#include <stdint.h>
void candidate_kernel(const uint8_t *in, uint8_t *out, int n,
                      const uint8_t *lut, int k){
    (void)k;
    for (int i = 0; i < n; i++) out[i] = lut[in[i]];
}
