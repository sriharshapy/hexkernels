/* Near-miss: wrapping signed difference a-b (no abs); wrong whenever a<b. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n){
  for(int i=0;i<n;i++) out[i] = (uint8_t)(a[i] - b[i]);
}
