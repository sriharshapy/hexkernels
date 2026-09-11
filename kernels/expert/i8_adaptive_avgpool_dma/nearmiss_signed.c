/* Near-miss: accumulates each window as SIGNED int8 (treats 128..255 as
 * -128..-1), so the per-window mean differs from the unsigned reference whenever
 * a window contains any byte > 127. */
#include <stdint.h>
void candidate_kernel(const uint8_t *in, int8_t *out, int n, int m){
  int k=n/m;
  for(int i=0;i<m;i++){
    int32_t s=0; for(int j=0;j<k;j++) s += (int32_t)(int8_t)in[i*k+j];
    out[i]=(int8_t)(s/k);
  }
}
