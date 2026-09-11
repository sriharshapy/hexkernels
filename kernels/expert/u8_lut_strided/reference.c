#include <stdint.h>
/* LUT applied to every k-th element; others are copied unchanged. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int n,
                      const uint8_t *lut, int k){
    for (int i = 0; i < n; i++){
        if (i % k == 0)
            out[i] = lut[in[i]];
        else
            out[i] = in[i];
    }
}
