/* Near-miss: nearest-neighbour (no interpolation) — ignores lo nibble fraction.
   Correct: out[i] = lut[hi] + (((lut[hi+1]-lut[hi])*lo) >> 4)
   This:    out[i] = lut[hi]   (wrong whenever lo != 0) */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut){
    for (int i = 0; i < n; i++){
        uint8_t b  = (uint8_t)in[i];
        int     hi = b >> 4;
        out[i] = lut[hi];
    }
}
