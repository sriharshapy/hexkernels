/* Near-miss: convolution (taps REVERSED) instead of correlation. Uses
 * taps[ntaps-1-j] so the result differs whenever the taps are not symmetric. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out, int n, int ntaps){
  for (int i=0;i<n;i++){
    int32_t acc=0;
    for (int j=0;j<ntaps;j++) acc += (int32_t)x[i+j]*(int32_t)taps[ntaps-1-j];
    out[i]=acc;
  }
}
