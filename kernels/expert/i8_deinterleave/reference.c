#include <stdint.h>
/* Deinterleave: a[i]=in[2*i], b[i]=in[2*i+1]. */
void candidate_kernel(const int8_t *in, int8_t *a, int8_t *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = in[2*i];
        b[i] = in[2*i + 1];
    }
}
