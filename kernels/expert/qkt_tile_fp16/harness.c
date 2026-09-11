#include "harness_common.h"
#include "kernel_api.h"

#define M  8
#define N  8
#define D  40   /* < 64 hf-lanes/vector -- zero-pad-tail path, single vector */

static hvx_hf Q[M*D] HVX_ALIGN;
static hvx_hf K[D*N] HVX_ALIGN;   /* column-major per key: K[d*N+j] */
static hvx_hf S[M*N] HVX_ALIGN;
static hvx_hf ref[M*N] HVX_ALIGN;

static float gen_hf(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 7) - 3;   /* -3..3 */
    return (float)(q * 0.25f);            /* exact in fp16 */
}

/* Independent scalar reference: same pinned formula, K indexed column-major. */
static void qkt_ref(const hvx_hf *Qv, const hvx_hf *Kv, hvx_hf *r,
                    int m, int n, int d, _Float16 scale) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            for (int k = 0; k < d; k++)
                acc += (float)Qv[i*d+k] * (float)Kv[k*n+j];
            hvx_hf mm = (hvx_hf)acc;                        /* hf-round #1 */
            r[i*n+j] = (hvx_hf)((float)mm * (float)scale);  /* hf-round #2 */
        }
    }
}

/* scale sweep: two distinct runtime values (anti-hardcode). */
static const float SCALES[] = { 0.125f, 1.0f };
#define NSETS 2

int main(void) {
    uint32_t s = 0x71E5A17u;

    for (int i = 0; i < M*D; i++) Q[i] = (hvx_hf)gen_hf(&s);
    for (int i = 0; i < D*N; i++) K[i] = (hvx_hf)gen_hf(&s);

    /* Edge cases: large positive product at d=0,j=0; sign-mismatch
     * cancellation at d=1; non-zero values right at the D-tail boundary
     * (d = D-2, D-1) to exercise the zero-pad-tail path with real data. */
    Q[0] = (hvx_hf)3.0f;      K[0] = (hvx_hf)3.0f;
    Q[D] = (hvx_hf)(-3.0f);   K[N] = (hvx_hf)3.0f;
    for (int i = 0; i < M; i++) { Q[i*D + (D-2)] = (hvx_hf)2.5f; Q[i*D + (D-1)] = (hvx_hf)(-1.5f); }
    for (int j = 0; j < N; j++) { K[(D-2)*N + j] = (hvx_hf)(-2.0f); K[(D-1)*N + j] = (hvx_hf)1.5f; }

    int errors = 0, fb = -1; long gotv = 0, expv = 0;

    for (int si = 0; si < NSETS; si++) {
        hvx_hf scale = (hvx_hf)SCALES[si];
        qkt_ref(Q, K, ref, M, N, D, scale);

        for (int i = 0; i < M*N; i++) *((volatile unsigned short *)&S[i]) = 0xA5A5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(Q, K, S, M, N, D, scale); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < M*N; i++) {
            unsigned short g = *(unsigned short *)&S[i], e = *(unsigned short *)&ref[i];
            if (!hvx_close_f16bits(g, e)) {
                errors++;
                if (fb < 0) { fb = si*M*N + i; gotv = (long)g; expv = (long)e; }
            }
        }
    }

    hvx_report_u16(errors, M*N*NSETS, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
