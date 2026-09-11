#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, const uint8_t *mask,
                      int8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = mask[i] ? a[i] : b[i];
}
