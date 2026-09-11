#include "harness_common.h"
#include "kernel_api.h"
#define M 8
#define N 8
#define K 94   /* 94 = 23*4 + 2 -- NOT a multiple of 4 (reduction tail) */

static int8_t A[M*K] HVX_ALIGN;
static int8_t B[K*N] HVX_ALIGN;
static int8_t out[M*N] HVX_ALIGN;
static int8_t ref[M*N] HVX_ALIGN;

/* Independent scalar reference: same pinned zero-point decomposition. */
static void zp_gemm_ref(const int8_t *Av, const int8_t *Bv, int8_t *r,
                        int m, int n, int k,
                        int32_t zpA, int32_t smult, int sshift) {
    for (int j = 0; j < n; j++) {
        int32_t colsum = 0;
        for (int kk = 0; kk < k; kk++) colsum += (int32_t)Bv[kk*n + j];
        for (int i = 0; i < m; i++) {
            int32_t rawdot = 0;
            for (int kk = 0; kk < k; kk++)
                rawdot += (int32_t)Av[i*k + kk] * (int32_t)Bv[kk*n + j];
            int32_t acc = rawdot - zpA * colsum;
            int64_t rv   = (int64_t)acc * (int64_t)smult;
            int64_t half = (sshift > 0) ? ((int64_t)1 << (sshift - 1)) : 0;
            int64_t q    = (rv >= 0) ? ((rv + half) >> sshift)
                                     : -((-rv + half) >> sshift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            r[i*n + j] = (int8_t)q;
        }
    }
}

/* Param sweep: (zpA, scale_mult, scale_shift) triples --
 * zpA=0 (degenerate: must reduce to plain GEMM requant), a NEGATIVE zpA,
 * and a POSITIVE zpA (with scale_shift=0, no rounding). */
static const int32_t ZPA[]     = {   0, -10,  20 };
static const int32_t SMULTS[]  = {   3,   5,   1 };
static const int     SSHIFTS[] = {   7,   4,   0 };
#define NSETS 3

int main(void) {
    uint32_t s = 0x3B77F210u;

    for (int i = 0; i < M*K; i++) A[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < K*N; i++) B[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    A[0] = 127;  B[0] = 127;              /* large positive contribution */
    A[1] = -128; B[N] = 127;              /* sign-mismatch cancellation */
    /* K tail: non-zero values at the last two k's (92, 93) */
    for (int i = 0; i < M; i++) { A[i*K+92] = 100; A[i*K+93] = -50; }
    for (int j = 0; j < N; j++) { B[92*N+j] = -80; B[93*N+j] = 60; }

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        int32_t zpA    = ZPA[p];
        int32_t smult  = SMULTS[p];
        int     sshift = SSHIFTS[p];

        zp_gemm_ref(A, B, ref, M, N, K, zpA, smult, sshift);

        for (int i = 0; i < M*N; i++) out[i] = (int8_t)0xA5;   /* poison */

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, out, M, N, K, zpA, smult, sshift); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < M*N; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) {
                    fb = p*M*N + i;
                    gotv = (long)out[i];
                    expv = (long)ref[i];
                }
            }
        }
    }

    hvx_report(errors, M*N*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
