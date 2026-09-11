/* Near-miss: plain ReLU (no upper cap at 6*scale). */
#include <stdint.h>
void candidate_kernel(const int8_t*x,int8_t*o,int n,int8_t scale){
  (void)scale; for(int i=0;i<n;i++){ int8_t v=x[i]; if(v<0)v=0; o[i]=v; }
}
