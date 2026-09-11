/* Scalar baseline: per-block dot product, exact int32 accumulation.
 * out[m] = sum_{k=0}^{blk-1} (int32)a[m*blk+k] * (int32)b[m*blk+k]
 * a is uint8 (activations), b is int8 (weights). */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, const int8_t *b, int32_t *out, int n, int blk) {
    int nb = n / blk;
    for (int m = 0; m < nb; m++) {
        int32_t s = 0;
        for (int k = 0; k < blk; k++)
            s += (int32_t)a[m*blk+k] * (int32_t)b[m*blk+k];
        out[m] = s;
    }
}
