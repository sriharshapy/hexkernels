/* layernorm_affine_i16 harness. Batched (R rows) int16 LayerNorm with
 * PER-ROW gamma[R]/beta[R] scalar affine (broadcast over columns) and the
 * inv-scale/gamma-scale epilogue order flipped vs. layernorm_row_i16.
 * Harness owns main() and computes the scalar reference INDEPENDENTLY
 * (same pinned formula). */
#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>

#define R 6
#define C 90    /* NOT a multiple of 64 int16-lanes-per-HVX-vector -- real tail */

static int16_t  x[R*C]      HVX_ALIGN;
static int16_t  out[R*C]    HVX_ALIGN;
static int16_t  ref[R*C]    HVX_ALIGN;
static int16_t  gamma_v[R]  HVX_ALIGN;
static int16_t  beta_v[R]   HVX_ALIGN;
static uint16_t inv_lut[256] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x2C6A11u;

    /* gamma/beta: per-ROW scalars, deterministic, guaranteed negative AND
     * positive values across the 6 rows. */
    for (int r = 0; r < R; r++) {
        int g = (r % 2 == 0) ? (10 + r) : -(10 + r);   /* alternating sign */
        if (g == 0) g = 1;
        gamma_v[r] = (int16_t)g;
        beta_v[r]  = (int16_t)((r - 3) * 300);          /* mix of +/-/0 */
    }

    /* inv_lut: a REAL non-degenerate monotonic-decreasing table, proportional
     * to 1/sqrt(var), built at runtime (never hardcoded inside candidate_kernel). */
    for (int i = 0; i < 256; i++) {
        double v = 4096.0 / sqrt((double)i + 1.0);
        inv_lut[i] = (uint16_t)(v + 0.5);
    }

    /* Row 0: all-identical values -> var=0 -> vidx=0 (LUT-clamp-low path). */
    for (int c = 0; c < C; c++) x[0*C + c] = -7;

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

        int32_t g = (int32_t)gamma_v[r];
        int32_t b = (int32_t)beta_v[r];

        for (int c = 0; c < C; c++) {
            int32_t d      = (int32_t)xr[c] - mu;
            int32_t normed = (d * inv + 512) >> 10;
            int32_t scaled = (normed * g + 32) >> 6;
            int32_t res    = scaled + b;
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
