#include <stdint.h>
/* 16-entry table, input reinterpreted as uint8, split into:
   hi nibble (bits 7..4) = table index (0..15)
   lo nibble (bits 3..0) = interpolation fraction (0..15)
   out[i] = lut[hi] + (((lut[hi+1] - lut[hi]) * lo) >> 4)   (truncating shift) */
void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut){
    for (int i = 0; i < n; i++){
        uint8_t b  = (uint8_t)in[i];
        int     hi = b >> 4;
        int     lo = b & 0xF;
        int16_t base  = lut[hi];
        int16_t delta = (int16_t)(lut[hi + 1] - lut[hi]);
        out[i] = (int8_t)(base + ((delta * lo) >> 4));
    }
}
