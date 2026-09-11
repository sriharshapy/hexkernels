/* Near-miss: indexes with (int)in[i]+128 (wrong offset) instead of (uint8_t)in[i]. */
#include <stdint.h>
void candidate_kernel(const int8_t*in,int8_t*o,int n,const int8_t*lut){
  for(int i=0;i<n;i++) o[i]=lut[(int)in[i]+128];
}
