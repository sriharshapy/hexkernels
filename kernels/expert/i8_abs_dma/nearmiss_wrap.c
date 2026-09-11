/* Near-miss: two's-complement abs WITHOUT -128 saturation (|-128| wraps to -128). */
#include <stdint.h>
void candidate_kernel(const int8_t*a,int8_t*o,int n){
  for(int i=0;i<n;i++){ int8_t v=a[i]; o[i]=(int8_t)(v<0?-v:v); }
}
