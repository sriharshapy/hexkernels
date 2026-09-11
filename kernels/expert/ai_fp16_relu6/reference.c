#include <stdint.h>
#include <math.h>
/* Correct scalar baseline: ReLU6 clamping in fp16.
 * out[i] = (hvx_hf)(fminf(fmaxf((float)x[i], 0.0f), 6.0f)) */
typedef __fp16 hvx_hf;
void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++) {
        float v = (float)x[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        out[i] = (hvx_hf)v;
    }
}
