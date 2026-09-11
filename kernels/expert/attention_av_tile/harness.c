#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define M  16
#define N  16
#define D  130   /* NOT a multiple of 128 */

static int8_t  A[M*N]  HVX_ALIGN;
static int8_t  V[N*D]  HVX_ALIGN;
static int32_t O[M*D]  HVX_ALIGN;
static int32_t ref[M*D] HVX_ALIGN;

/* Reference: same pinned formula */
static void av_ref(const int8_t *Av, const int8_t *Vv, int32_t *r,
                   int m, int n, int d) {
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < d; k++) {
            int32_t acc = 0;
            for (int j = 0; j < n; j++)
                acc += (int32_t)Av[i*n+j] * (int32_t)Vv[j*d+k];
            r[i*d+k] = acc;
        }
    }
}

int main(void) {
    uint32_t s = 0xFEEDCAFEu;

    /* Random inputs */
    for (int i = 0; i < M*N; i++) A[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < N*D; i++) V[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    A[0] = 127;  V[0] = 127;  /* large positive */
    A[N] = -128; V[D] = 127;  /* sign interaction */
    /* D tail: set non-zero values at [128,129] */
    for (int j = 0; j < N; j++) { V[j*D+128] = 100; V[j*D+129] = -50; }

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
