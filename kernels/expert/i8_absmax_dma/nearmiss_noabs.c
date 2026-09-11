/* Near-miss: reduces the signed max (no absolute value), so it misses the
 * injected -128 (magnitude 128) and returns a value <= 127. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int n, int32_t *out){
  int32_t m = -2147483647; for(int i=0;i<n;i++){ int32_t v=(int32_t)a[i]; if(v>m)m=v; } out[0]=m;
}
