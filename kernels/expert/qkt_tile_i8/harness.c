#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define M  16
#define N  16
#define D  98   /* NOT a multiple of 128 (tail when reducing in groups of 4) */

static int8_t  Q[M*D] HVX_ALIGN;
static int8_t  K[D*N] HVX_ALIGN;   /* column-major per key: K[d*N+j] */
static int8_t  S[M*N] HVX_ALIGN;
static int8_t  ref[M*N] HVX_ALIGN;

/* Independent scalar reference: same pinned formula, K indexed column-major. */
static void qkt_ref(const int8_t *Qv, const int8_t *Kv, int8_t *r,
                    int m, int n, int d,
                    int32_t smult, int sshift) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int k = 0; k < d; k++)
                acc += (int32_t)Qv[i*d+k] * (int32_t)Kv[k*n+j];
            int64_t rv   = (int64_t)acc * (int64_t)smult;
            int64_t half = (sshift > 0) ? ((int64_t)1 << (sshift - 1)) : 0;
            int64_t q    = (rv >= 0) ? ((rv + half) >> sshift)
                                     : -((-rv + half) >> sshift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            r[i*n+j] = (int8_t)q;
        }
    }
}

/*
 * Param sweep: different (scale_mult, scale_shift) pairs, including a
 * scale_shift=0 (no rounding) case.  Candidate MUST read params at
 * runtime (anti-hardcode).
 */
static const int32_t SMULTS[]  = { 3, 1, 5 };
static const int     SSHIFTS[] = { 7, 0, 3 };
#define NSETS 3

int main(void) {
    uint32_t s = 0xA17E4321u;

    /* Random Q and K */
    for (int i = 0; i < M*D; i++) Q[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < D*N; i++) K[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    Q[0] = 127;  K[0] = 127;              /* large positive dot at d=0,j=0 */
    Q[D] = -128; K[N] = 127;              /* sign-mismatch cancellation at d=1 */
    /* D tail: non-zero values in the tail group [96,97] (98 = 24*4 + 2) */
    for (int i = 0; i < M; i++) { Q[i*D+96] = 100; Q[i*D+97] = -50; }
    for (int j = 0; j < N; j++) { K[96*N+j] = -80; K[97*N+j] = 60; }

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int k = 0; k < NSETS; k++) {
        int32_t smult  = SMULTS[k];
        int     sshift = SSHIFTS[k];

        qkt_ref(Q, K, ref, M, N, D, smult, sshift);

        for (int i = 0; i < M*N; i++) S[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(Q, K, S, M, N, D, smult, sshift); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < M*N; i++) {
            if (S[i] != ref[i]) {
                errors++;
                if (fb < 0) {
                    fb = k * M*N + i;
                    gotv = (long)S[i];
                    expv = (long)ref[i];
                }
            }
        }
    }

    hvx_report(errors, M*N*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
