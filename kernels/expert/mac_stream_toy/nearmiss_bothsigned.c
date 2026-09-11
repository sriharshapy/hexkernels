/* Near-miss: treats a[] as SIGNED int8 instead of uint8 activations, so any
 * a[i] >= 128 contributes the wrong (negative) value to the block dot product. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, const int8_t *b, int32_t *out, int n, int blk){
  int nb = n / blk;
  for (int m = 0; m < nb; m++) {
    int32_t s = 0;
    for (int k = 0; k < blk; k++) s += (int32_t)(int8_t)a[m*blk+k] * (int32_t)b[m*blk+k];
    out[m] = s;
  }
}
