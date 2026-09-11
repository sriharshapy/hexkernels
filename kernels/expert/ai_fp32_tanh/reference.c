/* Correct scalar baseline: pointwise tanh activation, n=1024.
 * PINNED: tanh_f32(x) = tanhf(x), all fp32, no down-cast.
 * Same tanh_f32() snippet as harness reference -> bit-exact. */
#include <math.h>

/* Pinned tanh activation.  ALL computation in fp32; output is fp32 (no cast).
 * Semantics: tanh_f32(x) = tanhf(x)  (Hexagon libm).
 * #include <math.h> required.  Do NOT change the form or constant types. */
static float tanh_f32(float x) {
    return tanhf(x);
}

void candidate_kernel(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = tanh_f32(x[i]);
}
