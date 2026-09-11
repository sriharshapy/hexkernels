/* Near-miss: arithmetic-shift truncation (round toward -inf) instead of
 * round-half-away-from-zero; also matches on positives but differs on negatives. */
#include <stdint.h>
void candidate_kernel(const int8_t*a,int8_t*o,int n,int32_t scale,int32_t shift,int s){
  for(int i=0;i<n;i++){
    long v=(long)a[i]*scale+shift;
    long r=v>>s;               /* arithmetic shift, no round-half-away */
    if(r>127)r=127; if(r<-128)r=-128; o[i]=(int8_t)r;
  }
}
