#include "harness_common.h"
#include "kernel_api.h"
#define M  8    /* tokens */
#define K  16   /* FFN input dim */
#define V  32   /* FFN hidden dim */
#define D  16   /* FFN output dim = residual/LN dim */

static int8_t  x     [M*K]   HVX_ALIGN;
static int8_t  W1    [K*V]   HVX_ALIGN;
static int32_t b1    [V]     HVX_ALIGN;
static int8_t  gelu_lut[256] HVX_ALIGN;
static int8_t  W2    [V*D]   HVX_ALIGN;
static int32_t b2    [D]     HVX_ALIGN;
static int8_t  x_res [M*D]   HVX_ALIGN;
static int8_t  gamma_v[D]    HVX_ALIGN;
static int8_t  beta_v [D]    HVX_ALIGN;
static uint8_t inv_lut[256]  HVX_ALIGN;
static int8_t  out   [M*D]   HVX_ALIGN;
static int8_t  ref   [M*D]   HVX_ALIGN;

static int8_t sat8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static void build_gelu_lut(int8_t *lut, int variant) {
    for (int i = 0; i < 256; i++) {
        int v = i - 128;
        float fv = (float)v;
        if (variant == 1) fv *= 0.9f;
        float t = 0.7978845608f * (fv + 0.044715f * fv * fv * fv);
        float th;
        if      (t >  4.0f) th =  1.0f;
        else if (t < -4.0f) th = -1.0f;
        else { float x2 = t*t; th = t*(27.0f + x2)/(27.0f + 9.0f*x2); }
        float gv = 0.5f * fv * (1.0f + th);
        int r = (int)(gv >= 0 ? gv + 0.5f : gv - 0.5f);
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        lut[i] = (int8_t)r;
    }
}

/* Reference: full block for one param set */
static void compute_ref(void) {
    int8_t hid[M*V];

    /* Layer 1: GEMM + b1 + sat8 + GELU LUT */
    for (int i = 0; i < M; i++) {
        for (int v = 0; v < V; v++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)x[i*K+k] * (int32_t)W1[k*V+v];
            acc += b1[v];
            int8_t pre = sat8(acc);
            uint8_t idx = (uint8_t)(pre + 128);
            hid[i*V+v] = gelu_lut[idx];
        }
    }

    int8_t res[M*D];

    /* Layer 2: GEMM + b2 then residual add */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) {
            int32_t acc = 0;
            for (int v = 0; v < V; v++)
                acc += (int32_t)hid[i*V+v] * (int32_t)W2[v*D+j];
            int64_t ffn_out = (int64_t)acc + (int64_t)b2[j];
            /* Residual add then sat8 */
            int64_t added = ffn_out + (int32_t)x_res[i*D+j];
            res[i*D+j] = sat8((int32_t)added);
        }
    }

    /* LayerNorm per token */
    for (int i = 0; i < M; i++) {
        int32_t sum = 0;
        for (int j = 0; j < D; j++) sum += (int32_t)res[i*D+j];
        int32_t mu = sum / D;

        int32_t var_sum = 0;
        for (int j = 0; j < D; j++) {
            int32_t d = (int32_t)res[i*D+j] - mu;
            var_sum += d * d;
        }
        int32_t var = var_sum / D;
        int32_t v_idx = var;
        if (v_idx < 0)   v_idx = 0;
        if (v_idx > 255) v_idx = 255;
        uint8_t inv = inv_lut[v_idx];

        for (int j = 0; j < D; j++) {
            int32_t d      = (int32_t)res[i*D+j] - mu;
            int32_t scaled = (d * (int32_t)gamma_v[j] + 64) >> 7;
            int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
            int32_t r      = normed + (int32_t)beta_v[j];
            ref[i*D+j] = sat8(r);
        }
    }
}

/* Two full parameter sets (GELU LUT + LN params swept) */
#define NSETS 2

int main(void) {
    uint32_t s = 0xC7A1F3u;
    for (int i = 0; i < M*K; i++) x  [i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < K*V; i++) W1 [i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < V;   i++) b1 [i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < V*D; i++) W2 [i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < D;   i++) b2 [i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < M*D; i++) x_res[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Param arrays for two sweeps */
    int8_t  gammas[NSETS][D];
    int8_t  betas [NSETS][D];
    uint8_t luts  [NSETS][256];

    for (int p = 0; p < NSETS; p++) {
        for (int j = 0; j < D; j++) {
            gammas[p][j] = (int8_t)((hvx_lcg(&s) >> 24) | 1); /* non-zero */
            betas [p][j] = (int8_t)(hvx_lcg(&s) >> 24);
        }
        for (int k = 0; k < 256; k++)
            luts[p][k] = (uint8_t)(hvx_lcg(&s) >> 24);
        luts[p][0] = 180 + (uint8_t)(p * 15);
    }

    /* Edge cases */
    x[0] = 127; W1[0] = 127;    /* large product layer 1 */
    x[K] = -128; W1[V] = 127;   /* sign edge */
    b1[0] = -200000;             /* force negative GELU suppression */
    b1[1] =  200000;             /* force large hidden */
    x_res[0] = 127; x_res[1] = -128; /* large residual extremes */
    /* Plant a constant-x_res token row to exercise LN var=0 path */
    for (int j = 0; j < D; j++) x_res[(M-1)*D+j] = 5;

    int errors = 0, fb = -1; long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        build_gelu_lut(gelu_lut, p);
        for (int j = 0; j < D; j++) {
            gamma_v[j] = gammas[p][j];
            beta_v [j] = betas [p][j];
        }
        for (int k = 0; k < 256; k++) inv_lut[k] = luts[p][k];

        compute_ref();

        for (int i = 0; i < M*D; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, W1, b1, gelu_lut, W2, b2, x_res, gamma_v, beta_v, inv_lut, out, M, K, V, D); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < M*D; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = p*M*D + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
        }
    }
    hvx_report(errors, M*D*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
