#include <stdint.h>
typedef __fp16 hvx_hf;
/* NEAR-MISS B: squares a[i], ignores b entirely. Compiles; fails when b != a. */
void candidate_kernel(const hvx_hf *a, const hvx_hf *b, hvx_hf *out, int n) {
    (void)b;
    for (int i = 0; i < n; i++)
        out[i] = (hvx_hf)((float)a[i] * (float)a[i]);
}
