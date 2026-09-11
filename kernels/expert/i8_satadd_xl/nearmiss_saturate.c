/* Near-miss: two's-complement wraparound add instead of saturating add
 * (classic confusion with the sibling int8_vadd_xl task: 127+1 wraps to
 * -128 here, but the correct saturating result is 127). */
#include <stdint.h>
void candidate_kernel(const int8_t*a,const int8_t*b,int8_t*o,int n){
  for(int i=0;i<n;i++){int v=a[i]+b[i]; o[i]=(int8_t)v;}
}
