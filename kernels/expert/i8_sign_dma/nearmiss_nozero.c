/* Near-miss: two-way sign that maps 0 -> 1 (misses the three-way zero case). */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n){
  for(int i=0;i<n;i++) out[i] = (int8_t)(in[i] < 0 ? -1 : 1);
}
