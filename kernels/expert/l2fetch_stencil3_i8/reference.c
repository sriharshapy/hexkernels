/* Scalar baseline: 1D 3-tap weighted (1,2,1) stencil, edge-replicated,
 * saturating int8:
 *   out[i] = clamp(a[clamp(i-1,0,n-1)] + 2*a[i] + a[clamp(i+1,0,n-1)], -128, 127)
 */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int ip = (i > 0) ? i - 1 : 0;
        int in = (i < n - 1) ? i + 1 : n - 1;
        int32_t t = (int32_t)a[ip] + 2 * (int32_t)a[i] + (int32_t)a[in];
        if (t > 127) t = 127;
        if (t < -128) t = -128;
        out[i] = (int8_t)t;
    }
}
