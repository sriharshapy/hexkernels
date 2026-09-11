#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n){
    int8_t cur = -128;
    for (int i=0; i<n; i++){
        if (in[i] > cur) cur = in[i];
        out[i] = cur;
    }
}
