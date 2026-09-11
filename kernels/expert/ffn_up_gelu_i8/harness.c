#include "harness_common.h"
#include "kernel_api.h"

#define M 8
#define N 8
#define K 90   /* NOT a multiple of 4: 90 = 22*4 + 2 (2-element tail) */

/* acc max |val| = 127*127*90 = 1,451,430 -- easily within int32. */
static int8_t  A[M*K]        HVX_ALIGN;
static int8_t  B[N*K]        HVX_ALIGN;   /* B TRANSPOSED: row j contiguous over K */
static int32_t bias[N]       HVX_ALIGN;
static int8_t  gelu_lut[256] HVX_ALIGN;
static int8_t  out[M*N]      HVX_ALIGN;
static int8_t  ref[M*N]      HVX_ALIGN;

/* tanh via a rational Pade approximation (avoids a libm dependency in the
 * bare-metal sim -- same technique used by gemm_bias_gelu/i8_gemm_gelu_lut). */
static float tanh_approx(float t) {
    if (t >  4.0f) return  1.0f;
    if (t < -4.0f) return -1.0f;
    float x2 = t * t;
    return t * (27.0f + x2) / (27.0f + 9.0f * x2);
}

/* gelu_lut[idx] = clamp(round(v * sigmoid(1.702*v/32)), -128, 127), v = idx-128.
 * sigmoid(x) = 0.5*(1+tanh(x/2)). Two variants (scale on v) to prevent hardcoding. */
static void build_gelu_lut(int8_t *lut, int variant) {
    for (int i = 0; i < 256; i++) {
        float v  = (float)(i - 128);
        float vv = (variant == 1) ? v * 0.7f : v;
        float x  = 1.702f * vv / 32.0f;
        float th = tanh_approx(x * 0.5f);
        float sig = 0.5f * (1.0f + th);
        float g  = vv * sig;
        int r = (int)(g + (g >= 0 ? 0.5f : -0.5f));
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        lut[i] = (int8_t)r;
    }
}

#define NSETS 2

int main(void) {
    uint32_t s = 0xE17FA22u;
    for (int i = 0; i < M*K; i++) A[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < N*K; i++) B[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int j = 0; j < N;   j++) bias[j] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases */
    A[0*K+0] = 127; B[0*K+0] = 127;          /* large positive product at k=0 */
    A[1*K+0] = -128; B[1*K+0] = 127;         /* sign edge in reduction */
    /* K tail: nonzero values in the tail group [88,89] (90 = 22*4 + 2) */
    for (int i = 0; i < M; i++) { A[i*K+88] = 100; A[i*K+89] = -50; }
    for (int j = 0; j < N; j++) { B[j*K+88] = -80; B[j*K+89] = 60; }
    /* Force some columns' bias to saturate pre-LUT clamp in both directions */
    bias[0] = -2000000;   /* large negative bias -> pre clamps to -128 */
    bias[1] =  2000000;   /* large positive bias -> pre clamps to 127 */
    bias[2] = 0;          /* zero bias, activations both signs */

    int errors = 0, fb = -1; long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        build_gelu_lut(gelu_lut, p);

        /* Build reference */
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)A[i*K+k] * (int32_t)B[j*K+k];
                int64_t biased = (int64_t)acc + (int64_t)bias[j];
                if (biased >  127) biased =  127;
                if (biased < -128) biased = -128;
                int8_t pre = (int8_t)biased;
                uint8_t idx = (uint8_t)(pre + 128);
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
