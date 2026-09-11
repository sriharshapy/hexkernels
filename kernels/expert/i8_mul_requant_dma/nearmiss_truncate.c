/* Near-miss: arithmetic-shift truncation (round toward -inf) instead of
 * round-half-away-from-zero. Matches on positives that don't round up but differs
 * on negatives and on halfway cases. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp){
  for (int i=0;i<n;i++){
    long v = (long)((int32_t)a[i]*(int32_t)b[i]) * (long)mult;
    long r = v >> shift;         /* arithmetic shift, no round-half-away */
    r += zp; if (r>127) r=127; if (r<-128) r=-128; out[i]=(int8_t)r;
  }
}
