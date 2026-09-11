#include <stdint.h>
/* Scalar baseline: elementwise fp32 product.
 * This is the reference semantics the candidate must match bit-exactly. */
void candidate_kernel(const float *a, const float *b, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = a[i] * b[i];
}
