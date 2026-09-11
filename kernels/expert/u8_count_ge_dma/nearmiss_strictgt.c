/* Near-miss: uses STRICT '>' instead of inclusive '>='. Undercounts by the
 * number of elements equal to t (the harness injects several a[i]==t). */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, int n, uint8_t t, int32_t *out){
  int32_t s=0; for(int i=0;i<n;i++) if(a[i] > t) s++; out[0]=s;
}
