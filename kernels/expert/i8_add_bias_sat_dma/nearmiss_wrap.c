/* Near-miss: two's-complement wraparound add instead of int8 saturation.
 * Matches on non-overflowing elements but differs wherever x[i]+bias exceeds the
 * int8 range (e.g. 127+100 wraps to -29 instead of saturating to 127). */
#include <stdint.h>
void candidate_kernel(const int8_t *x, int8_t *out, int n, int32_t bias){
  for (int i=0;i<n;i++) out[i]=(int8_t)((int32_t)x[i]+bias);   /* wrap, no clamp */
}
