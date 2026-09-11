#include <stdint.h>
/* Scalar baseline: dual weighted-sum multiply-add, int16 wrap. */
void candidate_kernel(const uint8_t *a, const uint8_t *b,
                      const uint8_t *wa, const uint8_t *wb,
                      int16_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int16_t)((int)a[i]*(int)wa[i] + (int)b[i]*(int)wb[i]);
}
