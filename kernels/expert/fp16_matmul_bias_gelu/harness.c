#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N 32
#define KDIM 32

static hvx_hf A[N*KDIM] HVX_ALIGN, B[KDIM*N] HVX_ALIGN;
static hvx_hf bias[N] HVX_ALIGN;
static hvx_hf out[N*N] HVX_ALIGN, ref[N*N] HVX_ALIGN;

/* Pinned GELU tanh approximation -- identical constants in harness and baseline/candidate. */
static float gelu_f32(float x) {
    float c = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

static float gen_hf(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 13) - 6;   /* -6..6, both signs so gelu's branches are exercised */
    return (float)(q * 0.25f);             /* exact in fp16 */
}

int main(void) {
    uint32_t s = 0xF16Eu;
    for (int i = 0; i < N*KDIM; i++) A[i] = (hvx_hf)gen_hf(&s);
    for (int i = 0; i < KDIM*N; i++) B[i] = (hvx_hf)gen_hf(&s);

    /* Pinned bias: bias[j] = ((j%7)-3)*0.5f -> -1.5,-1,-0.5,0,0.5,1,1.5 cycling */
    for (int j = 0; j < N; j++) bias[j] = (hvx_hf)(((j % 7) - 3) * 0.5f);

    /* Independent scalar golden -- cross-checked against baseline.c/expert.c both. */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < KDIM; k++) acc += (float)A[i*KDIM+k] * (float)B[k*N+j];
            hvx_hf m = (hvx_hf)acc;                           /* fp16-round */
            float t = (float)m + (float)bias[j];              /* bias add in float */
            ref[i*N+j] = (hvx_hf)gelu_f32(t);                /* gelu then fp16 round */
        }

    for (int i = 0; i < N*N; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, bias, out, N, KDIM); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N*N; i++) {
        unsigned short g = *(unsigned short *)&out[i], e = *(unsigned short *)&ref[i];
        if (!hvx_close_f16bits(g, e)) {
            errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report_u16(errors, N*N, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
