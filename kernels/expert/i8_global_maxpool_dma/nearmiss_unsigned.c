/* Near-miss: takes the UNSIGNED-byte max (treats -128..-1 as 128..255), so it
 * returns the byte with the largest unsigned value (the injected -1 = 0xFF)
 * instead of the true signed max (+127). */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int n, int8_t *out){
  uint8_t m=0; for(int i=0;i<n;i++){ uint8_t v=(uint8_t)a[i]; if(v>m) m=v; } out[0]=(int8_t)m;
}
