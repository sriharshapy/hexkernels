/* Correct scalar baseline: pointwise SiLU (swish) activation, n=1024.
 * PINNED: silu_f32(x) = x / (1.0f + expf(-x)), all fp32, no down-cast.
 * Same silu_f32() snippet as harness reference -> bit-exact. */
#include <math.h>

/* Pinned SiLU (swish) activation.  ALL computation in fp32; output is fp32 (no cast).
 * Semantics: silu_f32(x) = x / (1.0f + expf(-x))  i.e. x * sigmoid(x).
 * #include <math.h> required.  Do NOT change the form or constant types. */
static float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

void candidate_kernel(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = silu_f32(x[i]);
}
