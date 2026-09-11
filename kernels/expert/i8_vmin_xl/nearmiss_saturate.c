/* Near-miss: unsigned byte min instead of signed int8 min (classic sign trap:
 * treats -1 (0xFF) as larger than 1, when signed min(-1,1) should be -1 but the
 * unsigned comparison picks 1 as the "smaller" value). */
#include <stdint.h>
void candidate_kernel(const int8_t*a,const int8_t*b,int8_t*o,int n){
  for(int i=0;i<n;i++){
    uint8_t ua=(uint8_t)a[i], ub=(uint8_t)b[i];
    o[i]=(int8_t)((ua<ub)?ua:ub);
  }
}
