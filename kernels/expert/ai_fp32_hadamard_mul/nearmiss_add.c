#include <stdint.h>
/* NEAR-MISS A: adds a[i]+b[i] instead of multiplying. Compiles; wrong result. */
void candidate_kernel(const float *a, const float *b, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}
