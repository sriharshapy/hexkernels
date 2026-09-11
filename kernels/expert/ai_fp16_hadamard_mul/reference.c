#include <stdint.h>
typedef __fp16 hvx_hf;
/* Scalar baseline: elementwise fp16 product via float multiply then cast.
 * This is the reference semantics the candidate must match bit-exactly. */
void candidate_kernel(const hvx_hf *a, const hvx_hf *b, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (hvx_hf)((float)a[i] * (float)b[i]);
}
