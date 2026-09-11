#include <stdint.h>
/* int16 index clamped to [0, lutsize-1], then lut lookup -> int8 out.
   Clamping prevents out-of-bounds access for negative or oversized indices. */
void candidate_kernel(const int16_t *in, int8_t *out, int n,
                      const int8_t *lut, int lutsize){
    for (int i = 0; i < n; i++){
        int idx = in[i];
        if (idx < 0) idx = 0;
        if (idx >= lutsize) idx = lutsize - 1;
        out[i] = lut[idx];
    }
}
