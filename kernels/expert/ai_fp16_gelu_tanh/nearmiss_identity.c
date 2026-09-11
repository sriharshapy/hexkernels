/* Near-miss B: identity (no activation).
 * Compiles and runs but fails because the reference applies GELU.
 * Fails on every element where gelu(x) != x (i.e., all non-large-positive x). */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = x[i];   /* identity — wrong, missing GELU */
}
