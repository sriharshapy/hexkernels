/* Scalar baseline: straight signum reading/writing DDR directly (no VTCM
 * staging). Correct scalar reference. This is the speedup denominator. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int8_t)(in[i] > 0 ? 1 : (in[i] < 0 ? -1 : 0));
}
