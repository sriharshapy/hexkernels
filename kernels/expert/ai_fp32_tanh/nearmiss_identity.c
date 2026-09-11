/* Near-miss B: identity (passthrough) instead of tanh.
 * Fails because tanh(x) != x for any |x| > 0:
 * tanh is bounded in (-1, 1) while identity is unbounded.
 * For x=0.125f: tanh(0.125)~0.1245 != 0.125 -> immediate mismatch.
 * Also fails at edge cases x=-6, x=6 (identity gives +-6, tanh gives ~+-1). */
#include <stdint.h>

void candidate_kernel(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = x[i];   /* identity, not tanh */
}
