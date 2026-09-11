#include <stdint.h>
/* Clamp the unsigned index to [lo,hi], then look up the 256-entry table. */
void candidate_kernel(const int8_t *in, int8_t *out, int n,
                      const int8_t *lut, uint8_t lo, uint8_t hi){
    for (int i = 0; i < n; i++){
        uint8_t idx = (uint8_t)in[i];
        if (idx < lo) idx = lo;
        if (idx > hi) idx = hi;
        out[i] = lut[idx];
    }
}
