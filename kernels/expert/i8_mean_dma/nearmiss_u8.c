/* Near-miss: sums as UNSIGNED bytes (treats -128..-1 as 128..255) before the
 * divide, so the mean differs from the signed int32 reference whenever any
 * a[i] < 0. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int n, int32_t *out){
  int32_t s=0; for(int i=0;i<n;i++) s += (int32_t)(uint8_t)a[i]; out[0]= s / n;
}
