/* Baseline: scalar per-channel PReLU reading/writing DDR directly (no VTCM
 * staging, no HVX). Passthrough for x>0, else arithmetic-shift-scaled;
 * saturating clamp to int8. Speedup denominator. */
#include <stdint.h>

void candidate_kernel(const int8_t *x, int8_t *out,
                      int n_ch, int n_elem,
                      const int8_t *alpha, int shift) {
    for (int c = 0; c < n_ch; c++) {
        int a = (int)alpha[c];
        const int8_t *xc = x + c*n_elem;
        int8_t *oc = out + c*n_elem;
        for (int i = 0; i < n_elem; i++) {
            int8_t v = xc[i];
            if (v > 0) { oc[i] = v; continue; }
            int r = ((int)v * a) >> shift;
            if (r > 127) r = 127; if (r < -128) r = -128;
            oc[i] = (int8_t)r;
        }
    }
}
