/* Near-miss: swaps a and b (selects b when mask!=0, a when mask==0).
   Fails at all positions where mask!=0 and a[i]!=b[i]. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, const uint8_t *mask,
                      int8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = mask[i] ? b[i] : a[i];
}
