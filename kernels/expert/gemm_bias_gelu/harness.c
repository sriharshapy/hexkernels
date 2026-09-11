#include "harness_common.h"
#include "kernel_api.h"
#define M 32
#define N 32
#define K 32

/* acc max |val| = 128*128*48 = 786,432 -- easily within int32.
 * After bias we saturate pre_lut to int8 and look up gelu_lut. */
static int8_t  A[M*K]      HVX_ALIGN;
static int8_t  B[K*N]      HVX_ALIGN;
static int32_t bias[N]     HVX_ALIGN;
static int8_t  gelu_lut[256] HVX_ALIGN;
static int8_t  out[M*N]    HVX_ALIGN;
static int8_t  ref[M*N]    HVX_ALIGN;

/* Construct a GELU-approximating LUT over quantized range.
 * GELU(x) ~ x * 0.5 * (1 + tanh(0.7978845608*(x + 0.044715*x^3))).
 * We use a fixed-point approximation stored as int8 output values. */
static void build_gelu_lut(int8_t *lut, int seed_variant) {
    /* Two variants driven by seed_variant to prevent hardcoding the LUT.
     * variant 0: standard GELU approximation scale 1.0
     * variant 1: scaled GELU (scale 0.75 -- shifts curve)
     * Both are valid nonlinear functions with negative inputs -> 0-ish outputs
     * and the lut is runtime-provided so candidates must use it. */
    for (int i = 0; i < 256; i++) {
        int v = i - 128;  /* signed value in [-128, 127] */
        /* Fixed-point GELU approximation: GELU(v) ~ v * sigmoid(1.702*v) */
        /* We use a simple piece-wise approximation that:
         * - for v < -3: returns ~0 (negative saturation)
         * - for v > 3:  returns ~v (identity)
         * - otherwise: smooth curve */
        float fv = (float)v;
        if (seed_variant == 1) fv *= 0.75f;
        /* approximate: 0.5*v*(1 + tanh(0.7978845608f * (fv + 0.044715f*fv*fv*fv))) */
        float t  = 0.7978845608f * (fv + 0.044715f*fv*fv*fv);
        /* tanh approximation: clamp at +-4 */
        float th;
        if (t >  4.0f) th =  1.0f;
        else if (t < -4.0f) th = -1.0f;
        else {
            /* tanh via polynomial: tanh(x) ~ x*(27+x^2)/(27+9*x^2) */
            float x2 = t*t;
            th = t*(27.0f + x2)/(27.0f + 9.0f*x2);
        }
        float gelu_val = 0.5f * fv * (1.0f + th);
        int r = (int)(gelu_val + (gelu_val >= 0 ? 0.5f : -0.5f));
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        lut[i] = (int8_t)r;
    }
}

/* We sweep two LUT variants to ensure candidates don't hardcode the LUT. */
#define NSETS 2

int main(void) {
    uint32_t s = 0xB7C4E1u;
    for (int i = 0; i < M*K; i++) A[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < K*N; i++) B[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < N;   i++) bias[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases: large products, negative bias (must hit GELU curve near 0), negative pre-LUT */
    A[0] = 127;  B[0] = 127;           /* large positive product */
    A[1] = -128; B[N] = 127;           /* sign edge in reduction */
    bias[0] = -2000000;                /* large negative bias -> pre_lut saturates to -128 */
    bias[1] =  2000000;                /* large positive bias -> pre_lut saturates to 127 */
    bias[2] = 0;                       /* zero bias, activations both signs */

    int errors = 0, fb = -1; long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        build_gelu_lut(gelu_lut, p);

        /* Build reference */
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
                int64_t biased = (int64_t)acc + (int64_t)bias[j];
                /* saturate to int8 */
                if (biased >  127) biased =  127;
                if (biased < -128) biased = -128;
                int8_t pre_lut = (int8_t)biased;
                uint8_t idx = (uint8_t)(pre_lut + 128);
                ref[i*N+j] = gelu_lut[idx];
            }
        }

        /* Poison output */
        for (int i = 0; i < M*N; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, bias, gelu_lut, out, M, N, K); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < M*N; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = p*M*N + i; gotv = (long)(unsigned char)out[i]; expv = (long)(unsigned char)ref[i]; }
            }
        }
    }
    hvx_report(errors, M*N*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
