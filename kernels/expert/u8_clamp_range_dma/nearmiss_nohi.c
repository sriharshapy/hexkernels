/* Near-miss: clamps only the lower bound (omits the hi clamp); wrong for in>191. */
#include <stdint.h>
#define LO 64
void candidate_kernel(const uint8_t *in, uint8_t *out, int n){
  for(int i=0;i<n;i++){ uint8_t x=in[i]; if(x<LO)x=LO; out[i]=x; }
}
