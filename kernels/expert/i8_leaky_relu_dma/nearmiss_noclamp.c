/* Near-miss: negative path without saturation clamp (wraps on overflow). */
#include <stdint.h>
void candidate_kernel(const int8_t*x,int8_t*o,int n,int alpha,int shift){
  for(int i=0;i<n;i++){ if(x[i]>0){o[i]=x[i];continue;} o[i]=(int8_t)(((int)x[i]*alpha)>>shift); }
}
