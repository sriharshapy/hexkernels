/* Near-miss: omits the +128 round-to-nearest before the >>8 (truncates instead).
 * Differs from the reference on every element whose fractional part rounds up
 * (e.g. a=255,b=0,alpha=127 -> 126 instead of 127). */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out,
                      int n, uint16_t alpha){
  for (int i=0;i<n;i++){
    int v = ((int)alpha*(int)a[i] + (256-(int)alpha)*(int)b[i]) >> 8;   /* no +128 */
    out[i] = (uint8_t)v;
  }
}
