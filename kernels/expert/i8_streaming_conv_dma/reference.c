/* Scalar baseline: 5-tap FIR (correlation, asymmetric taps) + round/shift
 * requant to int8, over DDR (no VTCM staging). Pure scalar reference;
 * speedup denominator. Same arithmetic as the generic loop below; the
 * ntaps==5 case (the only case this task's harness ever exercises) is
 * hand-unrolled to keep the functional-mode simulator within its wall-clock
 * budget for the large N used by this DDR-bound variant. */
#include <stdint.h>

static inline int8_t requant_ref(int32_t acc, int shift) {
    int32_t half = shift > 0 ? (1 << (shift - 1)) : 0;
    int32_t r = (acc >= 0) ? ((acc + half) >> shift) : -(((-acc) + half) >> shift);
    if (r > 127) r = 127; if (r < -128) r = -128;
    return (int8_t)r;
}

static inline int8_t fir5_ref(const int8_t * restrict xp, const int8_t * restrict taps,
                              int ntaps, int shift) {
    int32_t acc = 0;
    for (int j = 0; j < ntaps; j++) acc += (int32_t)xp[j] * (int32_t)taps[j];
    return requant_ref(acc, shift);
}

void candidate_kernel(const int8_t * restrict x, const int8_t * restrict taps,
                      int8_t * restrict out, int n, int ntaps, int shift) {
    if (ntaps == 5) {
        const int32_t t0 = taps[0], t1 = taps[1], t2 = taps[2], t3 = taps[3], t4 = taps[4];
        for (int i = 0; i < n; i++) {
            int32_t acc = (int32_t)x[i] * t0 + (int32_t)x[i+1] * t1 + (int32_t)x[i+2] * t2 +
                          (int32_t)x[i+3] * t3 + (int32_t)x[i+4] * t4;
            out[i] = requant_ref(acc, shift);
        }
        return;
    }
    for (int i = 0; i < n; i++) out[i] = fir5_ref(x + i, taps, ntaps, shift);
}
