/* Near-miss: averages the bytes as SIGNED int8 (treats 128..255 as -128..-1), so
 * the mean differs from the unsigned reference whenever any a[i] > 127. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, int n, uint8_t *out){
  int32_t s=0; for(int i=0;i<n;i++) s += (int32_t)(int8_t)a[i]; out[0]=(uint8_t)(s/n);
}
