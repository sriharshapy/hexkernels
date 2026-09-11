/* Near-miss: unsigned compare instead of signed. For c=0 this returns in[i]
 * unchanged for every byte (all bytes >= 0 unsigned), so negative inputs are
 * never clamped to 0 -> differs from the signed max wherever in[i] < 0. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n, int8_t c){
  uint8_t uc = (uint8_t)c;
  for (int i=0;i<n;i++){ uint8_t x=(uint8_t)in[i]; out[i]=(int8_t)((x>uc)?x:uc); }
}
