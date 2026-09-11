/* Near-miss: uses abs() (mirrors negatives) instead of ReLU (clamps to 0).
 * Matches on x>=0 but differs on every negative input. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, int8_t *out, int n){
  for(int i=0;i<n;i++){ int v=x[i]; out[i]=(int8_t)(v<0?-v:v); }
}
