/* Near-miss: saturating add instead of two's-complement wraparound. */
#include <stdint.h>
void candidate_kernel(const int8_t*a,const int8_t*b,int8_t*o,int n){
  for(int i=0;i<n;i++){int v=a[i]+b[i]; if(v>127)v=127; if(v<-128)v=-128; o[i]=(int8_t)v;}
}
