/* Near-miss: accumulates the SIGNED difference a[i]-b[i] instead of |a[i]-b[i]|.
 * Wrong whenever a[i] < b[i] for some i (cancellation / negative terms). */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, const uint8_t *b, int n, int32_t *out){
  int32_t s=0; for(int i=0;i<n;i++) s += (int)a[i] - (int)b[i]; out[0]=s;
}
