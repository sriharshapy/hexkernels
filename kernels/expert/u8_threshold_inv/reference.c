#include <stdint.h>
void candidate_kernel(const uint8_t *in, uint8_t *out, int n, uint8_t thresh) {
    for (int i = 0; i < n; i++) out[i] = (in[i] < thresh) ? 255 : 0;
}
