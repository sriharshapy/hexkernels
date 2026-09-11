#include "harness_common.h"
#include "kernel_api.h"
#define N 32

static hvx_hf A[N*N] HVX_ALIGN, B[N*N] HVX_ALIGN, out[N*N] HVX_ALIGN, ref[N*N] HVX_ALIGN;

static float gen_hf(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 7) - 3;   /* -3..3 */
    return (float)(q * 0.25f);           /* exact in fp16, |acc| bounded */
}

int main(void) {
    uint32_t s = 0x3AC1u;
    for (int i = 0; i < N*N; i++) A[i] = (hvx_hf)gen_hf(&s);
    for (int i = 0; i < N*N; i++) B[i] = (hvx_hf)gen_hf(&s);
    /* C0 accumulator preload (small ints, exact in fp16) */
    for (int i = 0; i < N*N; i++) {
        hvx_hf c0 = (hvx_hf)((int)(hvx_lcg(&s) % 9) - 4);   /* -4..4 */
        out[i] = c0; ref[i] = c0;
    }

    /* scalar reference: ref = C0 + A*B (float accumulate, cast to fp16) */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float acc = (float)ref[i*N+j];
            for (int k = 0; k < N; k++) acc += (float)A[i*N+k] * (float)B[k*N+j];
            ref[i*N+j] = (hvx_hf)acc;
        }

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, out, N); });
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
