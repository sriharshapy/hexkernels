#include <stdint.h>
/* NEAR-MISS B: squares a[i], ignores b entirely. Compiles; fails when b != a. */
void candidate_kernel(const float *a, const float *b, float *out, int n) {
    (void)b;
    for (int i = 0; i < n; i++)
        out[i] = a[i] * a[i];
}
