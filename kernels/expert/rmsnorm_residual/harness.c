#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define N 113   /* NOT a multiple of 128 */

static int8_t  x[N]         HVX_ALIGN;
static int8_t  res[N]       HVX_ALIGN;
static int8_t  gamma_v[N]   HVX_ALIGN;
static int8_t  out[N]       HVX_ALIGN;
static int8_t  ref[N]       HVX_ALIGN;
static uint8_t inv_lut[256] HVX_ALIGN;

/*
 * Reference: fused residual add + rmsnorm (pinned formula, SHIFT=8, NO mean sub).
 */
static void rmsnorm_residual_ref(const int8_t *xv, const int8_t *rv, int8_t *r, int n,
                                  const int8_t *gam, const uint8_t *lut) {
    /* Step 1: saturating residual add */
    int8_t t[N];
    for (int i = 0; i < n; i++) {
        int32_t s = (int32_t)xv[i] + (int32_t)rv[i];
        if (s >  127) s =  127;
        if (s < -128) s = -128;
        t[i] = (int8_t)s;
    }

    /* Step 2a: rms2 */
    int32_t sum_sq = 0;
    for (int i = 0; i < n; i++) {
        int32_t ti = (int32_t)t[i];
        sum_sq += ti * ti;
    }
    int32_t rms2 = sum_sq / n;

    /* Step 2b-c: LUT lookup */
    int32_t r_idx = rms2;
    if (r_idx < 0)   r_idx = 0;
    if (r_idx > 255) r_idx = 255;
    uint8_t inv = lut[(int)r_idx];

    /* Step 2d-f: per-element scale */
    for (int i = 0; i < n; i++) {
        int32_t ti     = (int32_t)t[i];
        int32_t scaled = (ti * (int32_t)gam[i] + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        if (normed >  127) normed =  127;
        if (normed < -128) normed = -128;
        r[i] = (int8_t)normed;
    }
}

/* Param sweep: 3 sets of (gamma, inv_lut) — anti-hardcode. */
#define NSETS 3

int main(void) {
    uint32_t s = 0xBEEF4567u;

    int8_t  gammas[NSETS][N];
    uint8_t luts  [NSETS][256];

    for (int k = 0; k < NSETS; k++) {
        for (int i = 0; i < N; i++) {
            int8_t g = (int8_t)(hvx_lcg(&s) >> 24);
            gammas[k][i] = (g == 0) ? 1 : g;  /* non-zero gamma */
        }
        for (int j = 0; j < 256; j++)
            luts[k][j] = (uint8_t)((hvx_lcg(&s) >> 24) & 0xFF);
        luts[k][0] = 215 + (uint8_t)(k * 10);  /* large inv when rms2=0 */
    }

    /* Generate inputs */
    for (int i = 0; i < N; i++) x[i]   = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < N; i++) res[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    x[0] = 127;  res[0] = 1;    /* saturating add at max */
    x[1] = -128; res[1] = -1;   /* saturating add at min */
    x[2] = 0;    res[2] = 0;
    /* zero segment: rms2=0, tests r_idx=0 path */
    for (int i = N-4; i < N; i++) { x[i] = 0; res[i] = 0; }

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int k = 0; k < NSETS; k++) {
        for (int i = 0; i < N; i++) gamma_v[i] = gammas[k][i];
        for (int j = 0; j < 256; j++) inv_lut[j] = luts[k][j];

        rmsnorm_residual_ref(x, res, ref, N, gamma_v, inv_lut);

        for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, res, out, N, gamma_v, inv_lut); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < N; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) {
                    fb = k * N + i;
                    gotv = (long)out[i];
                    expv = (long)ref[i];
                }
            }
        }
    }

    hvx_report(errors, N * NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
