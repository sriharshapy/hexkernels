/* Near-miss: divides the window sum by 8 with an arithmetic RIGHT SHIFT (floor
 * toward -inf) instead of C integer division (truncate toward zero). The two
 * disagree whenever the window sum is negative and not a multiple of 8 (the
 * harness injects negative extremes and a sign-varying ramp). */
#include <stdint.h>
void candidate_kernel(const int8_t *x, int8_t *out, int n, int W){
  (void)W;
  for (int i=0;i<n;i++){
    int32_t acc=0; for (int j=0;j<8;j++) acc += (int32_t)x[i+j];
    out[i] = (int8_t)(acc >> 3);   /* floor division, not truncation */
  }
}
