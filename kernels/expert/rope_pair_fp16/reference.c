/* PLAIN SCALAR RoPE-on-pairs -- the DENOMINATOR baseline. No HVX
 * intrinsics, no vector types anywhere. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *x, const hvx_hf *cos, const hvx_hf *sin,
                      hvx_hf *out, int n_pairs) {
    for (int p = 0; p < n_pairs; p++) {
        int d = 2*p, d2 = 2*p + 1;
        float xr = (float)x[d], xi = (float)x[d2];
        float c = (float)cos[p], s = (float)sin[p];
        out[d]  = (hvx_hf)(xr * c - xi * s);
        out[d2] = (hvx_hf)(xr * s + xi * c);
    }
}
