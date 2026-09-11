#include "harness_common.h"
#include "kernel_api.h"

#define M  8
#define N  40  /* reduction axis, < 64 hf-lanes/vector -- zero-pad-tail path */
#define D  8

static hvx_hf A[M*N] HVX_ALIGN;
static hvx_hf V[D*N] HVX_ALIGN;   /* column-major: V[d*N+j] */
static hvx_hf O[M*D] HVX_ALIGN;
static hvx_hf ref[M*D] HVX_ALIGN;

static float gen_hf(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 7) - 3;   /* -3..3 */
    return (float)(q * 0.25f);            /* exact in fp16 */
}

/* Independent scalar reference: same pinned formula, V column-major. */
static void av_ref(const hvx_hf *Av, const hvx_hf *Vv, hvx_hf *r,
                   int m, int n, int d) {
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < d; k++) {
            float acc = 0.f;
            for (int j = 0; j < n; j++)
                acc += (float)Av[i*n+j] * (float)Vv[k*n+j];
            r[i*d+k] = (hvx_hf)acc;
        }
    }
}

int main(void) {
    uint32_t s = 0xC0FFEE11u;

    for (int i = 0; i < M*N; i++) A[i] = (hvx_hf)gen_hf(&s);
    for (int i = 0; i < D*N; i++) V[i] = (hvx_hf)gen_hf(&s);

    /* Edge cases: large positive product at j=0,d=0; sign-mismatch
     * cancellation at j=1; non-zero values right at the N-tail boundary
     * (j=N-2,N-1) to exercise the zero-pad-tail path with real data. */
    A[0] = (hvx_hf)3.0f;   V[0] = (hvx_hf)3.0f;
    A[1] = (hvx_hf)(-3.0f); V[N+1] = (hvx_hf)3.0f;
    for (int i = 0; i < M; i++) { A[i*N + (N-2)] = (hvx_hf)2.5f; A[i*N + (N-1)] = (hvx_hf)(-1.5f); }
    for (int k = 0; k < D; k++) { V[k*N + (N-2)] = (hvx_hf)(-2.0f); V[k*N + (N-1)] = (hvx_hf)1.5f; }

    av_ref(A, V, ref, M, N, D);

    for (int i = 0; i < M*D; i++) *((volatile unsigned short *)&O[i]) = 0xA5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, V, O, M, N, D); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < M*D; i++) {
        unsigned short g = *(unsigned short *)&O[i], e = *(unsigned short *)&ref[i];
        if (!hvx_close_f16bits(g, e)) {
            errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report_u16(errors, M*D, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
