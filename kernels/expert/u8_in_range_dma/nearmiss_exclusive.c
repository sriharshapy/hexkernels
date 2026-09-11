/* Near-miss: exclusive bounds (wrong at in==64 and in==191, which must be 255). */
#include <stdint.h>
#define LO 64
#define HI 191
void candidate_kernel(const uint8_t *in, uint8_t *out, int n){
  for(int i=0;i<n;i++) out[i] = (uint8_t)((in[i] > LO && in[i] < HI) ? 255 : 0);
}
