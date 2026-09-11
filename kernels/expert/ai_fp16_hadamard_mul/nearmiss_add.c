#include <stdint.h>
typedef __fp16 hvx_hf;
/* NEAR-MISS A: adds a[i]+b[i] instead of multiplying. Compiles; wrong result. */
void candidate_kernel(const hvx_hf *a, const hvx_hf *b, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (hvx_hf)((float)a[i] + (float)b[i]);
}
