/* Baseline: pure scalar NCHW -> NHWC interleave. For C=4 channel planes,
 * gathers out[p*C + c] = in[c*HW + p] for every pixel p. Pure data movement,
 * no arithmetic. This is the speedup denominator. */
#include <stdint.h>

void candidate_kernel(const int8_t *in, int8_t *out, int C, int HW) {
    for (int p = 0; p < HW; p++)
        for (int c = 0; c < C; c++)
            out[p*C + c] = in[c*HW + p];
}
