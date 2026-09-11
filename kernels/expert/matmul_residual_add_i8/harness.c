#include "harness_common.h"
#include "kernel_api.h"
#define M 8
#define N 8
#define K 90   /* 90 = 22*4 + 2 -- NOT a multiple of 4 (reduction tail) */

static int8_t A[M*K]        HVX_ALIGN;
static int8_t B[K*N]        HVX_ALIGN;
static int8_t residual[M*N] HVX_ALIGN;
static int8_t out[M*N]      HVX_ALIGN;
static int8_t ref[M*N]      HVX_ALIGN;

/* Independent scalar reference: residual added AFTER requantize/shift. */
static void matres_ref(const int8_t *Av, const int8_t *Bv, const int8_t *Rv,
                       int8_t *r, int m, int n, int k,
                       int32_t smult, int sshift) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int kk = 0; kk < k; kk++)
                acc += (int32_t)Av[i*k + kk] * (int32_t)Bv[kk*n + j];
            int64_t rv   = (int64_t)acc * (int64_t)smult;
            int64_t half = (sshift > 0) ? ((int64_t)1 << (sshift - 1)) : 0;
            int64_t q    = (rv >= 0) ? ((rv + half) >> sshift)
                                     : -((-rv + half) >> sshift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            int32_t requantized = (int32_t)q;

            int32_t sum = requantized + (int32_t)Rv[i*n + j];
            if (sum >  127) sum =  127;
            if (sum < -128) sum = -128;
            r[i*n + j] = (int8_t)sum;
        }
    }
}

/* Param sweep: >=2 distinct (scale_mult, scale_shift) pairs. */
static const int32_t SMULTS[]  = { 3, 1 };
static const int     SSHIFTS[] = { 6, 0 };
#define NSETS 2

int main(void) {
    uint32_t s = 0x9A4C21E7u;

    for (int i = 0; i < M*K; i++) A[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < K*N; i++) B[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < M*N; i++) residual[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    A[0] = 127;  B[0] = 127;              /* large positive contribution */
    A[1] = -128; B[N] = 127;              /* sign-mismatch cancellation */
    /* K tail: non-zero values at the last two k's (88, 89) */
    for (int i = 0; i < M; i++) { A[i*K+88] = 100; A[i*K+89] = -50; }
    for (int j = 0; j < N; j++) { B[88*N+j] = -80; B[89*N+j] = 60; }
    /* Residual saturation edges: max acc contribution paired with +/-127 residual */
    residual[0] = 127;    /* row0,col0: requantized will be large positive -> saturate */
    residual[N+1] = -128; /* row1,col1 */

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        int32_t smult  = SMULTS[p];
        int     sshift = SSHIFTS[p];

        matres_ref(A, B, residual, ref, M, N, K, smult, sshift);

        for (int i = 0; i < M*N; i++) out[i] = (int8_t)0xA5;   /* poison */

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, residual, out, M, N, K, smult, sshift); });
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
