/* Near-miss: computes the MIN instead of the max, so it differs from the
 * reference on any non-constant input. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, int n, uint8_t *out){
  uint8_t m=255; for(int i=0;i<n;i++) if(a[i]<m)m=a[i]; out[0]=m;
}
