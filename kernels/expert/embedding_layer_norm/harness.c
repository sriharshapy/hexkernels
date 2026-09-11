#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define VOCAB  256
#define D       64   /* embedding dim */
#define T       32   /* token count, NOT a multiple of 128 */

static int8_t  table[VOCAB * D] HVX_ALIGN;
static int32_t idx[T]           HVX_ALIGN;
static int8_t  gamma_v[D]       HVX_ALIGN;
static int8_t  beta_v[D]        HVX_ALIGN;
static int8_t  out[T * D]       HVX_ALIGN;
static int8_t  ref[T * D]       HVX_ALIGN;
static uint8_t inv_lut[256]     HVX_ALIGN;

/*
 * Reference: gather then layernorm per token.
 */
static void embed_ln_ref(const int8_t *tab, const int32_t *id, int8_t *r,
                          int t, int d,
                          const int8_t *gam, const int8_t *bet,
                          const uint8_t *lut) {
    for (int i = 0; i < t; i++) {
        const int8_t *row = tab + (int)id[i] * d;
        int8_t *out_row   = r + i * d;

        /* LayerNorm on gathered row */
        int32_t sum = 0;
        for (int j = 0; j < d; j++) sum += (int32_t)row[j];
        int32_t mu = sum / d;

        int32_t var_sum = 0;
        for (int j = 0; j < d; j++) {
            int32_t dv = (int32_t)row[j] - mu;
            var_sum += dv * dv;
        }
        int32_t var = var_sum / d;
        int32_t v_idx = var;
        if (v_idx < 0)   v_idx = 0;
        if (v_idx > 255) v_idx = 255;
        uint8_t inv = lut[(int)v_idx];

        for (int j = 0; j < d; j++) {
            int32_t dv     = (int32_t)row[j] - mu;
            int32_t scaled = (dv * (int32_t)gam[j] + 64) >> 7;
            int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
            int32_t res    = normed + (int32_t)bet[j];
            if (res >  127) res =  127;
            if (res < -128) res = -128;
            out_row[j] = (int8_t)res;
        }
    }
}

/*
 * Sweep 2 index arrays + 3 (gamma, beta, inv_lut) parameter sets.
 * Anti-hardcode: must read table, idx, gamma, beta, inv_lut at runtime.
 */
#define NIDX  2
#define NPARAMS 3

int main(void) {
    uint32_t s = 0xFACE5678u;

    /* Generate table */
    for (int i = 0; i < VOCAB * D; i++)
        table[i] = (int8_t)(hvx_lcg(&s) >> 24);

    int32_t idxs[NIDX][T];
    for (int k = 0; k < NIDX; k++) {
        for (int i = 0; i < T; i++)
            idxs[k][i] = (int32_t)((hvx_lcg(&s) >> 24) % VOCAB);
    }

    int8_t  gammas[NPARAMS][D];
    int8_t  betas [NPARAMS][D];
    uint8_t luts  [NPARAMS][256];

    for (int k = 0; k < NPARAMS; k++) {
        for (int j = 0; j < D; j++) {
            int8_t g = (int8_t)(hvx_lcg(&s) >> 24);
            gammas[k][j] = (g == 0) ? 1 : g;
            betas[k][j]  = (int8_t)(hvx_lcg(&s) >> 24);
        }
        for (int j = 0; j < 256; j++)
            luts[k][j] = (uint8_t)((hvx_lcg(&s) >> 24) & 0xFF);
        luts[k][0] = 200 + (uint8_t)(k * 15);
    }

    /* Edge cases for index arrays */
    idxs[0][0]   = 0;         /* first vocab row */
    idxs[0][1]   = VOCAB - 1; /* last vocab row */
    idxs[0][2]   = 0;         /* repeated index */
    idxs[0][T-1] = VOCAB / 2;
    idxs[1][0]   = VOCAB - 1;
    idxs[1][T-1] = 0;

    /* Constant embedding row to exercise var=0 path */
    for (int d = 0; d < D; d++) table[0 * D + d] = 5;  /* vocab row 0 constant */
    idxs[0][3] = 0;  /* one token points to the constant row */
    idxs[1][3] = 0;

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int ki = 0; ki < NIDX; ki++) {
        for (int i = 0; i < T; i++) idx[i] = idxs[ki][i];

        for (int kp = 0; kp < NPARAMS; kp++) {
            for (int j = 0; j < D; j++) {
                gamma_v[j] = gammas[kp][j];
                beta_v[j]  = betas[kp][j];
            }
            for (int j = 0; j < 256; j++) inv_lut[j] = luts[kp][j];

            embed_ln_ref(table, idx, ref, T, D, gamma_v, beta_v, inv_lut);

            for (int i = 0; i < T * D; i++) out[i] = (int8_t)0xA5;

            unsigned long long _hvx_kc = 0;
            HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(table, idx, out, T, D, gamma_v, beta_v, inv_lut); });
            printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

            for (int i = 0; i < T * D; i++) {
                if (out[i] != ref[i]) {
                    errors++;
                    if (fb < 0) {
                        fb = ki * NPARAMS * T * D + kp * T * D + i;
                        gotv = (long)out[i];
                        expv = (long)ref[i];
                    }
                }
            }
        }
    }

    hvx_report(errors, NIDX * NPARAMS * T * D, fb, gotv, expv);
    return errors ? 1 : 0;
}
