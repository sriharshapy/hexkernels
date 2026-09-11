#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>
#include <string.h>

#define N      128   /* total elements; multiple of G */
#define G      4     /* number of chunks */
#define CHUNK  (N/G) /* 32 elements per chunk */

static int8_t  x[N]         HVX_ALIGN;
static int8_t  gamma_v[N]   HVX_ALIGN;
static int8_t  beta_v[N]    HVX_ALIGN;
static int8_t  out[N]       HVX_ALIGN;
static int8_t  ref[N]       HVX_ALIGN;
static uint8_t inv_lut[256] HVX_ALIGN;

/* Reference: identical pinned formula, per-chunk */
static void chunk_norm_ref(const int8_t *xv, int8_t *r, int n, int g,
                           const int8_t *gam, const int8_t *bet,
                           const uint8_t *lut) {
    int chunk = n / g;
    for (int c = 0; c < g; c++) {
        const int8_t *xc = xv + c * chunk;
        int8_t       *rc = r  + c * chunk;
        const int8_t *gc = gam + c * chunk;
        const int8_t *bc = bet + c * chunk;

        /* Step 1: mean */
        int32_t sum = 0;
        for (int j = 0; j < chunk; j++) sum += (int32_t)xc[j];
        int32_t mu = sum / chunk;

        /* Step 2: variance */
        int32_t var_sum = 0;
        for (int j = 0; j < chunk; j++) {
            int32_t d = (int32_t)xc[j] - mu;
            var_sum += d * d;
        }
        int32_t var = var_sum / chunk;

        /* Step 3-4: inv_lut lookup */
        int32_t v_idx = var;
        if (v_idx < 0)   v_idx = 0;
        if (v_idx > 255) v_idx = 255;
        uint8_t inv = lut[(int)v_idx];

        /* Step 5: per-element scale+offset */
        for (int j = 0; j < chunk; j++) {
            int32_t d      = (int32_t)xc[j] - mu;
            int32_t scaled = (d * (int32_t)gc[j] + 64) >> 7;
            int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
            int32_t res    = normed + (int32_t)bc[j];
            if (res >  127) res =  127;
            if (res < -128) res = -128;
            rc[j] = (int8_t)res;
        }
    }
}

/*
 * Param sweep: 3 distinct (gamma, beta, inv_lut) sets.
 * Candidate must read all runtime pointers (anti-hardcode).
 */
#define NSETS 3

int main(void) {
    uint32_t s = 0xC4E21A07u;

    int8_t  gammas[NSETS][N];
    int8_t  betas [NSETS][N];
    uint8_t luts  [NSETS][256];

    for (int k = 0; k < NSETS; k++) {
        for (int i = 0; i < N; i++) {
            gammas[k][i] = (int8_t)((hvx_lcg(&s) >> 24) | 1); /* non-zero */
            betas [k][i] = (int8_t)(hvx_lcg(&s) >> 24);
        }
        for (int j = 0; j < 256; j++)
            luts[k][j] = (uint8_t)((hvx_lcg(&s) >> 24) & 0xFF);
        luts[k][0] = 200 + (uint8_t)(k * 15);  /* large inv when var=0 */
    }

    /* Generate input */
    for (int i = 0; i < N; i++) x[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    /* Chunk 0: constant value -> var=0 (tests lut[0] path) */
    for (int j = 0; j < CHUNK; j++) x[0*CHUNK + j] = 42;
    /* Chunk 1: extremes -> large variance */
    x[1*CHUNK + 0] = 127;
    x[1*CHUNK + 1] = -128;
    /* Chunk 2: all zeros */
    for (int j = 0; j < CHUNK; j++) x[2*CHUNK + j] = 0;
    /* Chunk 3: alternating +/- */
    for (int j = 0; j < CHUNK; j++) x[3*CHUNK + j] = (j & 1) ? -64 : 64;

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int k = 0; k < NSETS; k++) {
        for (int i = 0; i < N; i++) {
            gamma_v[i] = gammas[k][i];
            beta_v[i]  = betas[k][i];
        }
        for (int j = 0; j < 256; j++) inv_lut[j] = luts[k][j];

        chunk_norm_ref(x, ref, N, G, gamma_v, beta_v, inv_lut);

        for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N, G, gamma_v, beta_v, inv_lut); });
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
