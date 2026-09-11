#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define M  16
#define N  16
#define D  90   /* NOT a multiple of 32 -- output-loop tail */

static int8_t  A[M*N]  HVX_ALIGN;
static int8_t  V[D*N]  HVX_ALIGN;   /* column-major: V[d*N+j] */
static int32_t O[M*D]  HVX_ALIGN;
static int32_t ref[M*D] HVX_ALIGN;

/* Independent scalar reference: same pinned formula, V column-major. */
static void av_ref(const int8_t *Av, const int8_t *Vv, int32_t *r,
                   int m, int n, int d) {
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < d; k++) {
            int32_t acc = 0;
            for (int j = 0; j < n; j++)
                acc += (int32_t)Av[i*n+j] * (int32_t)Vv[k*n+j];
            r[i*d+k] = acc;
        }
    }
}

int main(void) {
    uint32_t s = 0xC0FFEE11u;

    /* Random inputs */
    for (int i = 0; i < M*N; i++) A[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < D*N; i++) V[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    A[0] = 127;  V[0] = 127;   /* large positive, d=0,j=0 */
    A[N] = 0;    /* keep row 1 clean */
    A[1] = -128; V[N+1] = 127; /* sign interaction, d=1,j=1 */

    av_ref(A, V, ref, M, N, D);

    for (int i = 0; i < M*D; i++) O[i] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, V, O, M, N, D); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;
    for (int i = 0; i < M*D; i++) {
        if (O[i] != ref[i]) {
            errors++;
            if (fb < 0) {
                fb = i;
                gotv = (long)O[i];
                expv = (long)ref[i];
            }
        }
    }

    hvx_report(errors, M*D, fb, gotv, expv);
    return errors ? 1 : 0;
}
