#include <stdint.h>
void candidate_kernel(const uint8_t *in, uint8_t *out, int n, const uint8_t *lut){
    for (int i = 0; i < n; i++) out[i] = lut[in[i]];
}
