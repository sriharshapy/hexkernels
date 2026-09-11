#include <stdint.h>
/* Reverse: out[i] = in[n-1-i]. */
void candidate_kernel(const int8_t *in, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = in[n - 1 - i];
}
