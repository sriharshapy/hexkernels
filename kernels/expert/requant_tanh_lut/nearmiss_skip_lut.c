/* Near-miss: skips the LUT lookup -- outputs the raw quantized index as int8.
   Always wrong because the LUT transforms the index to tanh-scaled output. */
#include <stdint.h>
void candidate_kernel(const int32_t *a, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp,
                      const int8_t lut[256]) {
    (void)lut;  /* BUG: LUT ignored */
    for (int i = 0; i < n; i++) {
        int64_t v    = (int64_t)a[i] * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += zp;
        if (r < 0)   r = 0;
        if (r > 255) r = 255;
        /* BUG: returns quantized index directly instead of lut[idx] */
        out[i] = (int8_t)(uint8_t)r;
    }
}
