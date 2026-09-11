/* i8_layernorm_dma harness (v6, group H holdout: hvx + dma + vtcm).
 * Row-wise integer LayerNorm over R=8192 rows of W=128 int8 (in+out > 2MB
 * >> L2 -> DDR-bandwidth-bound). Harness owns main(): maps VTCM identity,
 * seeds deterministic per-row-varying inputs (plus constant-row and
 * extreme-row edge cases so per-row stats genuinely differ row to row),
 * computes the pinned-formula scalar reference directly (this IS the
 * reference, not a derived golden -- no cross-check needed), poisons out,
 * times the candidate (kernel-only pcycles), bit-exact compares. */
#include <stddef.h>
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef R
#define R 4224
#endif
#define W 128

static int8_t  x[(size_t)R * W]   HVX_ALIGN;
static int8_t  out[(size_t)R * W] HVX_ALIGN;
static int8_t  ref[(size_t)R * W] HVX_ALIGN;
static int8_t  gamma_v[W]  HVX_ALIGN;
static int8_t  beta_v[W]   HVX_ALIGN;
static uint8_t inv_lut[256] HVX_ALIGN;

static void layernorm_row_ref(const int8_t *xr, int8_t *outr, int w,
                              const int8_t *gamma, const int8_t *beta,
                              const uint8_t *lut) {
    int32_t sum = 0;
    for (int i = 0; i < w; i++) sum += (int32_t)xr[i];
    int32_t mu = sum / w;

    int32_t var_sum = 0;
    for (int i = 0; i < w; i++) {
        int32_t d = (int32_t)xr[i] - mu;
        var_sum += d * d;
    }
    int32_t var = var_sum / w;
    int32_t v_idx = var;
    if (v_idx < 0) v_idx = 0;
    if (v_idx > 255) v_idx = 255;
    uint8_t inv = lut[(int)v_idx];

    for (int i = 0; i < w; i++) {
        int32_t d      = (int32_t)xr[i] - mu;
        int32_t scaled = (d * (int32_t)gamma[i] + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        int32_t r      = normed + (int32_t)beta[i];
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        outr[i] = (int8_t)r;
    }
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0xC0FFEE11u;

    /* Per-row-varying input: fully random per element -> distinct per-row
       mean/variance almost everywhere (so the global-stats near-miss and any
       cross-row confusion genuinely differ from the correct per-row result). */
    for (size_t i = 0; i < (size_t)R * W; i++) x[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* gamma/beta: per-column, non-zero gamma. inv_lut: 256 entries. */
    for (int i = 0; i < W; i++) {
        gamma_v[i] = (int8_t)((hvx_lcg(&s) >> 24) | 1);
        beta_v[i]  = (int8_t)(hvx_lcg(&s) >> 24);
    }
    for (int j = 0; j < 256; j++) inv_lut[j] = (uint8_t)((hvx_lcg(&s) >> 24) & 0xFF);
    inv_lut[0] = 220;   /* large inv when var=0 (constant row) */

    /* Edge-case rows: constant (var=0), extremes (max variance), first/last row. */
    for (int i = 0; i < W; i++) x[0 * W + i] = 5;                      /* row 0: constant, var=0 */
    for (int i = 0; i < W; i++) x[1 * W + i] = (int8_t)((i & 1) ? 127 : -128); /* row 1: max variance */
    for (int i = 0; i < W; i++) x[(size_t)(R - 1) * W + i] = (int8_t)((i & 1) ? -128 : 127); /* last row */

    for (int r = 0; r < R; r++)
        layernorm_row_ref(x + (size_t)r * W, ref + (size_t)r * W, W, gamma_v, beta_v, inv_lut);

    for (size_t i = 0; i < (size_t)R * W; i++) out[i] = (int8_t)0xA5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, R, W, gamma_v, beta_v, inv_lut); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    long errors = 0; long fb = -1; long gotv = 0, expv = 0;
    for (size_t i = 0; i < (size_t)R * W; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) { fb = (long)i; gotv = (long)out[i]; expv = (long)ref[i]; }
        }
    }
    hvx_report((int)errors, R * W, (int)fb, gotv, expv);
    return errors ? 1 : 0;
}
