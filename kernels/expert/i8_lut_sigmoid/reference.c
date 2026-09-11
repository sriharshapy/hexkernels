#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut){
    for (int i=0;i<n;i++) out[i] = lut[(uint8_t)in[i]];
}
