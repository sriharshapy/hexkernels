/* Near-miss: sign error on the FIRST (real) rotation component --
 * computes out[d] = x[d]*cos[p] + x[d2]*sin[p] instead of
 * x[d]*cos[p] - x[d2]*sin[p]. Compiles, "looks like" RoPE, but flips the
 * sign of the cross term. Fails broadly whenever sin[p]!=0, and in
 * particular fails the mandated cos=0,sin=1 edge pair: correct
 * out[d] = -x[d2], this produces out[d] = +x[d2]. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *x, const hvx_hf *cos, const hvx_hf *sin,
                      hvx_hf *out, int n_pairs) {
    for (int p = 0; p < n_pairs; p++) {
        int d = 2*p, d2 = 2*p + 1;
        float xr = (float)x[d], xi = (float)x[d2];
        float c = (float)cos[p], s = (float)sin[p];
        out[d]  = (hvx_hf)(xr * c + xi * s);   /* BUG: should be - */
        out[d2] = (hvx_hf)(xr * s + xi * c);
    }
}
