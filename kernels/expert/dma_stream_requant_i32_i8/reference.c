/* Baseline: scalar sign-aware requant reading a[] directly from DDR and
 * writing out[] directly to DDR (no HVX, no VTCM staging). Speedup
 * denominator. */
#include <stdint.h>

void candidate_kernel(const int32_t *a, int8_t *out, int n, int32_t mult, int shift, int8_t zp) {
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    for (int i = 0; i < n; i++) {
        int64_t v = a[i];
        int64_t absv = v < 0 ? -v : v;
        int64_t am = absv * (int64_t)mult;
        int64_t sh = (am + half) >> shift;
        int64_t r = (v < 0) ? -sh : sh;
        r += zp;
        if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
