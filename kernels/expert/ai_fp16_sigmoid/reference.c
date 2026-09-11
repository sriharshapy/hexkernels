/* Correct scalar baseline: pointwise sigmoid, n=1024.
 * PINNED: all intermediates fp32, single fp16 round at end.
 * Definition: sigmoid(x) = 1.0f / (1.0f + expf(-x))
 * Same sigmoid_f32() as harness reference -> bit-exact. */
#include "kernel_api.h"
#include <math.h>

/* Pinned sigmoid.  ALL intermediates in fp32; ONE fp16 round at end.
 * Definition: sigmoid(x) = 1.0f / (1.0f + expf(-x))
 * Do NOT change the formula or intermediate types. */
static float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (hvx_hf)sigmoid_f32((float)x[i]);
}
