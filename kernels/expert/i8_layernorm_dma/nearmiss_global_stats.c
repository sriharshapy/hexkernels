/* Near-miss: computes ONE mean/variance pooled across ALL R*W elements
 * (as if it were one giant row) instead of per-row statistics, then applies
 * that single (mu, inv) pair to every row. Plausible (a LayerNorm-shaped
 * bug: forgetting to reset the accumulator per row), but wrong whenever
 * rows have different statistics -- which the harness's rows do. */
#include <stdint.h>
#include <stddef.h>
void candidate_kernel(const int8_t *x, int8_t *out, int R, int W,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    long n = (long)R * (long)W;
    int64_t sum = 0;
    for (long i = 0; i < n; i++) sum += x[i];
    int32_t mu = (int32_t)(sum / n);

    int64_t var_sum = 0;
    for (long i = 0; i < n; i++) {
        int32_t d = (int32_t)x[i] - mu;
        var_sum += (int64_t)d * d;
    }
    int32_t var = (int32_t)(var_sum / n);
    int32_t v_idx = var;
    if (v_idx < 0) v_idx = 0;
    if (v_idx > 255) v_idx = 255;
    uint8_t inv = inv_lut[(int)v_idx];

    for (int r = 0; r < R; r++) {
        for (int i = 0; i < W; i++) {
            int32_t d      = (int32_t)x[(size_t)r * W + i] - mu;
            int32_t scaled = (d * (int32_t)gamma[i] + 64) >> 7;
            int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
            int32_t v      = normed + (int32_t)beta[i];
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            out[(size_t)r * W + i] = (int8_t)v;
        }
    }
}
