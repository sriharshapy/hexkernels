/* Baseline: scalar 3-tap conv1d, edge-replicated boundaries, bias+shift
 * requant (no HVX, no VTCM tiling/halo staging at all). Speedup denominator. */
#include <stdint.h>

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void candidate_kernel(const int8_t *a, const int8_t *w, int32_t bias, int shift,
                      int8_t *out, int n) {
    int32_t half = (shift > 0) ? (1 << (shift - 1)) : 0;
    for (int i = 0; i < n; i++) {
        int32_t left   = a[clampi(i - 1, 0, n - 1)];
        int32_t center = a[i];
        int32_t right  = a[clampi(i + 1, 0, n - 1)];
        int32_t t = (int32_t)w[0]*left + (int32_t)w[1]*center + (int32_t)w[2]*right + bias;
        int64_t abst = t < 0 ? -(int64_t)t : (int64_t)t;
        int64_t sh = (abst + half) >> shift;
        int64_t r = t < 0 ? -sh : sh;
        if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
