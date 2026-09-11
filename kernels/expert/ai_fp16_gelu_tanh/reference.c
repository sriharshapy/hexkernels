/* Correct scalar baseline: pointwise GELU (tanh approximation), n=1024.
 * PINNED: all intermediates fp32, single fp16 round at end.
 * Constants: 0.7978845608f = sqrt(2/pi), 0.044715f = standard coefficient.
 * Same gelu_f32() as harness reference -> bit-exact. */
#include "kernel_api.h"
#include <math.h>

/* Pinned GELU tanh approximation.  ALL intermediates in fp32; ONE fp16 round at end.
 * Constants: 0.7978845608 = sqrt(2/pi), 0.044715 = standard tanh-GELU coefficient.
 * Do NOT change the constant values or intermediate types. */
static float gelu_f32(float x) {
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (hvx_hf)gelu_f32((float)x[i]);
}
