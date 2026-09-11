/* Near-miss: accumulates a[i]+b[i] instead of a[i]*b[i]. */
#include <stdint.h>
void candidate_kernel(const int16_t *a, const int16_t *b, int n, int32_t *out){
  int32_t s=0; for(int i=0;i<n;i++) s += (int32_t)a[i] + (int32_t)b[i]; out[0]=s;
}
