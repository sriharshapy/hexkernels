#include "harness_common.h"
#include "kernel_api.h"
#define N 64

static uint8_t A[N*N]        HVX_ALIGN;
static int8_t  B[N*N]        HVX_ALIGN;
static int32_t bias[N]       HVX_ALIGN;
static int8_t  gelu_lut[256] HVX_ALIGN;
static int8_t  out[N*N]      HVX_ALIGN;
static int8_t  ref[N*N]      HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

static inline int8_t saturate_i8(int v) {
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int8_t)v;
}

/* GELU-approximating LUT over the quantized int8 pre-activation range.
 * GELU(x) ~ x * 0.5 * (1 + tanh(0.7978845608*(x + 0.044715*x^3))), same idiom
 * as datasets/v5/tasks/gemm_bias_gelu. Runtime-built so candidates cannot
 * hardcode it. */
static void build_gelu_lut(int8_t *lut) {
    for (int i = 0; i < 256; i++) {
        int v = i - 128;
        float fv = (float)v;
        float t = 0.7978845608f * (fv + 0.044715f*fv*fv*fv);
        float th;
        if (t > 4.0f) th = 1.0f;
        else if (t < -4.0f) th = -1.0f;
        else { float x2 = t*t; th = t*(27.0f + x2)/(27.0f + 9.0f*x2); }
        float gelu_val = 0.5f * fv * (1.0f + th);
        int r = (int)(gelu_val + (gelu_val >= 0 ? 0.5f : -0.5f));
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        lut[i] = (int8_t)r;
    }
}

int main(void) {
    uint32_t s = 0x73C4u;
    /* Same matmul domain as i8_matmul_hmx_64x64: A in 0..7, B in -3..3, so
     * |acc|<=64*7*3=1344 -> |r|<=1428. Bias is small (+-40) so most of the
     * int8-domain LUT gets exercised, plus explicit large edges to force
     * saturation at both ends and a zero-bias column. */
    for (int i = 0; i < N*N; i++) A[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < N*N; i++) B[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);
    for (int j = 0; j < N;   j++) bias[j] = (int32_t)((int)(hvx_lcg(&s) % 81) - 40);
    bias[0] = -3000; bias[1] = 3000; bias[2] = 0; /* force sat-low, sat-high, zero */

    build_gelu_lut(gelu_lut);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int acc = 0;
            for (int k = 0; k < N; k++) acc += (int)A[i*N+k] * (int)B[k*N+j];
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[j];
            int8_t pre_lut = saturate_i8(biased);
            uint8_t idx = (uint8_t)(pre_lut + 128);
            ref[i*N+j] = gelu_lut[idx];
        }

    for (int i = 0; i < N*N; i++) *((volatile signed char *)&out[i]) = (signed char)0x55; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, bias, gelu_lut, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N*N; i++) {
        int g = (int)out[i], e = (int)ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = g; expv = e; } }
    }
    hvx_report(errors, N*N, fb, gotv, expv);
    return errors ? 1 : 0;
}
