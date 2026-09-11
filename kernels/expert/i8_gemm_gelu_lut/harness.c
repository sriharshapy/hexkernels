#include "harness_common.h"
#include "kernel_api.h"
#define M 32
#define N 32
#define K 32

static int8_t  A[M*K]        HVX_ALIGN;
static int8_t  B[K*N]        HVX_ALIGN;
static int8_t  gelu_lut[256] HVX_ALIGN;
static int8_t  out[M*N]      HVX_ALIGN;
static int8_t  ref[M*N]      HVX_ALIGN;

/* Build GELU LUT variant (same as gemm_bias_gelu but no bias here). */
static void build_gelu_lut(int8_t *lut, int seed_variant) {
    for (int i = 0; i < 256; i++) {
        int v = i - 128;
        float fv = (float)v;
        if (seed_variant == 1) fv *= 0.75f;
        float t = 0.7978845608f * (fv + 0.044715f*fv*fv*fv);
        float th;
        if (t >  4.0f) th =  1.0f;
        else if (t < -4.0f) th = -1.0f;
        else {
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

/* Two LUT variants to prevent hardcoding. */
#define NSETS 2

int main(void) {
    uint32_t s = 0xD5F3A8u;
    for (int i = 0; i < M*K; i++) A[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < K*N; i++) B[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases: large products (saturate acc before LUT), negative pre-activation */
    A[0] = 127;  B[0] = 127;     /* large positive -> acc saturates high */
    A[1] = -128; B[N] = 127;     /* sign edge */
    /* Force acc to be large negative (many -128*+127 products) to hit GELU near 0 */
    for (int k = 0; k < K && k < 32; k++) {
        A[2*K+k] = -128;
        B[k*N+2] = 127;
    }

    int errors = 0, fb = -1; long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        build_gelu_lut(gelu_lut, p);

        /* Build reference */
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
                /* saturate to int8 */
                int32_t sat = acc;
                if (sat >  127) sat =  127;
                if (sat < -128) sat = -128;
                int8_t pre_lut = (int8_t)sat;
                uint8_t idx = (uint8_t)(pre_lut + 128);
                ref[i*N+j] = gelu_lut[idx];
            }
        }

        /* Poison output */
        for (int i = 0; i < M*N; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, gelu_lut, out, M, N, K); });
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
