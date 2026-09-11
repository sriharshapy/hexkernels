#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define H_DIM  16
#define W_DIM  16
#define C_CHANS 16
#define HW     (H_DIM * W_DIM)   /* 256 */
#define N      (C_CHANS * HW)    /* 4096 */

static int8_t  x[N]                  HVX_ALIGN;
static int8_t  gamma_v[C_CHANS]      HVX_ALIGN;
static int8_t  beta_v[C_CHANS]       HVX_ALIGN;
static int8_t  out[N]                HVX_ALIGN;
static int8_t  ref[N]                HVX_ALIGN;
/* C separate 256-entry LUTs */
static uint8_t inv_lut[C_CHANS * 256] HVX_ALIGN;

/*
 * Reference: instance normalization (per-channel, pinned formula, SHIFT=8).
 */
static void instance_norm_ref(const int8_t *xv, int8_t *r,
                               int H, int W, int C,
                               const int8_t *gam, const int8_t *bet,
                               const uint8_t *lut) {
    int hw = H * W;
    for (int c = 0; c < C; c++) {
        const uint8_t *clut = lut + c * 256;
        int base = c * hw;

        /* Step 1: mean */
        int32_t sum = 0;
        for (int s = 0; s < hw; s++) sum += (int32_t)xv[base + s];
        int32_t mu = sum / hw;

        /* Step 2: variance */
        int32_t var_sum = 0;
        for (int s = 0; s < hw; s++) {
            int32_t d = (int32_t)xv[base + s] - mu;
            var_sum += d * d;
        }
        int32_t var = var_sum / hw;

        /* Step 3-4: LUT lookup */
        int32_t v_idx = var;
        if (v_idx < 0)   v_idx = 0;
        if (v_idx > 255) v_idx = 255;
        uint8_t inv = clut[(int)v_idx];

        /* Step 5: per-element normalize + affine */
        for (int s = 0; s < hw; s++) {
            int pos    = base + s;
            int32_t d  = (int32_t)xv[pos] - mu;
            int32_t sc = (d * (int32_t)gam[c] + 64) >> 7;
            int32_t nm = (sc * (int32_t)inv + 128) >> 8;
            int32_t res = nm + (int32_t)bet[c];
            if (res >  127) res =  127;
            if (res < -128) res = -128;
            r[pos] = (int8_t)res;
        }
    }
}

/* Param sweep: 2 sets of (gamma, beta, inv_lut) — anti-hardcode. */
#define NSETS 2

int main(void) {
    uint32_t s = 0xDEAD9876u;

    int8_t  gammas[NSETS][C_CHANS];
    int8_t  betas [NSETS][C_CHANS];
    uint8_t luts  [NSETS][C_CHANS * 256];

    for (int k = 0; k < NSETS; k++) {
        for (int c = 0; c < C_CHANS; c++) {
            int8_t g = (int8_t)(hvx_lcg(&s) >> 24);
            gammas[k][c] = (g == 0) ? 1 : g;
            betas[k][c]  = (int8_t)(hvx_lcg(&s) >> 24);
        }
        for (int i = 0; i < C_CHANS * 256; i++)
            luts[k][i] = (uint8_t)((hvx_lcg(&s) >> 24) & 0xFF);
        /* Large inv when var=0 for each channel */
        for (int c = 0; c < C_CHANS; c++)
            luts[k][c * 256] = 200 + (uint8_t)(k * 10 + c * 2);
    }

    /* Generate inputs */
    for (int i = 0; i < N; i++) x[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    x[0] = 127; x[1] = -128;   /* extremes in channel 0 */
    x[2] = 0;   x[3] = 0;
    /* Constant channel: all elements of channel 15 = same value -> var=0 */
    for (int s2 = 0; s2 < HW; s2++) x[15 * HW + s2] = 42;

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int k = 0; k < NSETS; k++) {
        for (int c = 0; c < C_CHANS; c++) {
            gamma_v[c] = gammas[k][c];
            beta_v[c]  = betas[k][c];
        }
        for (int i = 0; i < C_CHANS * 256; i++) inv_lut[i] = luts[k][i];

        instance_norm_ref(x, ref, H_DIM, W_DIM, C_CHANS, gamma_v, beta_v, inv_lut);

        for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, H_DIM, W_DIM, C_CHANS, gamma_v, beta_v, inv_lut); });
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
