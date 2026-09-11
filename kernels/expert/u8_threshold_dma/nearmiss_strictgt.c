/* Near-miss: strict > instead of inclusive >= (wrong at in==128). */
#include <stdint.h>
#define THRESH 128
void candidate_kernel(const uint8_t *in, uint8_t *out, int n){
  for(int i=0;i<n;i++) out[i] = (uint8_t)(in[i] > THRESH ? 255 : 0);
}
