/* layernorm_row_i16 harness. Batched (R rows) int16 LayerNorm, LUT-based
 * inverse-std, no float/division-beyond-mean-var. Harness owns main() and
 * computes the scalar reference INDEPENDENTLY (same pinned formula). */
#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>

#define R 6
#define C 100   /* NOT a multiple of 64 int16-lanes-per-HVX-vector -- real tail */

static int16_t  x[R*C]     HVX_ALIGN;
static int16_t  out[R*C]   HVX_ALIGN;
static int16_t  ref[R*C]   HVX_ALIGN;
static int16_t  gamma_v[C] HVX_ALIGN;
static int16_t  beta_v[C]  HVX_ALIGN;
static uint16_t inv_lut[256] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x1A5E11u;

    /* gamma/beta: deterministic, guaranteed negative AND positive values. */
    for (int c = 0; c < C; c++) {
        int g = (c % 61) - 30;         /* [-30, 30] */
        if (g == 0) g = 1;             /* never zero */
        gamma_v[c] = (int16_t)g;
        beta_v[c] = (int16_t)(((c % 51) - 25) * 40);  /* [-1000, 1000] */
    }

    /* inv_lut: a REAL non-degenerate monotonic-decreasing table, proportional
     * to 1/sqrt(var), built at runtime (never hardcoded inside candidate_kernel). */
    for (int i = 0; i < 256; i++) {
        double v = 4096.0 / sqrt((double)i + 1.0);
        inv_lut[i] = (uint16_t)(v + 0.5);
    }

    /* Row 0: all-identical values -> var=0 -> vidx=0 (LUT-clamp-low path). */
    for (int c = 0; c < C; c++) x[0*C + c] = 42;

    /* Row 1: large spread -> high variance -> vidx clamps to 255. */
    for (int c = 0; c < C; c++) x[1*C + c] = (c % 2 == 0) ? 20000 : -20000;

    /* Rows 2..5: general random data (moderate range so mu/var stay well
     * within int16 headroom for every row). */
    for (int r = 2; r < R; r++)
        for (int c = 0; c < C; c++)
            x[r*C + c] = (int16_t)((int32_t)(hvx_lcg(&s) >> 24) % 101 - 50); /* [-50,50] */

    /* Independent scalar reference (same pinned formula, duplicated here --
     * never calls baseline.c/expert.c). */
    for (int r = 0; r < R; r++) {
        const int16_t *xr = x + (long)r * C;
        int16_t *refr = ref + (long)r * C;

        int64_t sum = 0;
        for (int c = 0; c < C; c++) sum += (int64_t)xr[c];
        int32_t mu = (int32_t)(sum / C);

        int64_t var_sum = 0;
        for (int c = 0; c < C; c++) {
            int64_t d = (int64_t)xr[c] - mu;
            var_sum += d * d;
        }
        int64_t var = var_sum / C;
        int32_t vidx = (int32_t)(var >> 5);
        if (vidx < 0) vidx = 0;
        if (vidx > 255) vidx = 255;
        int32_t inv = (int32_t)inv_lut[vidx];

        for (int c = 0; c < C; c++) {
            int32_t d      = (int32_t)xr[c] - mu;
            int32_t scaled = (d * (int32_t)gamma_v[c] + 32) >> 6;
            int32_t normed = (scaled * inv + 512) >> 10;
            int32_t res    = normed + (int32_t)beta_v[c];
            if (res >  32767) res =  32767;
            if (res < -32768) res = -32768;
            refr[c] = (int16_t)res;
        }
    }

    for (int i = 0; i < R*C; i++) *((volatile uint16_t *)&out[i]) = 0xA5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, R, C, gamma_v, beta_v, inv_lut); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < R*C; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)out[i]; expv = (long)ref[i]; }
        }
    }
    hvx_report(errors, R*C, fb, gotv, expv);
    return errors ? 1 : 0;
}
