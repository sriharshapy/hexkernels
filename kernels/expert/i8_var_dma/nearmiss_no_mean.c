/* Near-miss: returns the mean of squares sum(a^2)/n (forgets to subtract the
 * squared mean), so it differs from the variance whenever the mean is nonzero. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int n, int32_t *out){
  int64_t sumsq=0; for(int i=0;i<n;i++){ int64_t v=(int64_t)a[i]; sumsq += v*v; }
  out[0]=(int32_t)(sumsq/(int64_t)n);
}
