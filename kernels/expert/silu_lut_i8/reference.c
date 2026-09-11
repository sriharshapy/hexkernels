#include <stdint.h>
/* Direct-indexed 256-entry byte LUT gather: idx = (uint8_t)x[i]; out[i] = lut[idx]. */
void candidate_kernel(const int8_t *x, int8_t *out, int n, const int8_t *lut) {
    for (int i = 0; i < n; i++) {
        uint8_t idx = (uint8_t)x[i];
        out[i] = lut[idx];
    }
}
