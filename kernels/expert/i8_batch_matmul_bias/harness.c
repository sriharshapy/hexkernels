#include "harness_common.h"
#include "kernel_api.h"
#define M 16
#define N 16
#define K 32

/* BATCH defined in kernel_api.h = 4.
 * acc max |val| = 128*128*32 = 524,288; bias up to 16383; total fits int32. */
static int8_t  A[BATCH*M*K]       HVX_ALIGN;
static int8_t  B[BATCH*K*N]       HVX_ALIGN;
static int32_t bias[BATCH*N]      HVX_ALIGN;
static int32_t C[BATCH*M*N]       HVX_ALIGN;
static int32_t ref[BATCH*M*N]     HVX_ALIGN;

int main(void) {
    uint32_t s = 0xF2A3C4u;
    for (int i = 0; i < BATCH*M*K; i++) A[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < BATCH*K*N; i++) B[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < BATCH*N;   i++) bias[i]  = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases: distinct data per batch; large products; bias affects every row */
    A[0] = 127; B[0] = 127;              /* batch 0: large positive */
    A[M*K] = -128; B[K*N+N] = 127;      /* batch 1: sign edge */
    bias[0] = -200000;                   /* large negative bias[0], batch 0 */
    bias[N] =  200000;                   /* large positive bias[0], batch 1 */

    /* Golden reference */
    for (int b = 0; b < BATCH; b++) {
        const int8_t *Ab   = A    + b * M * K;
        const int8_t *Bb   = B    + b * K * N;
        const int32_t *bib = bias + b * N;
        int32_t      *rb   = ref  + b * M * N;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)Ab[i*K+k] * (int32_t)Bb[k*N+j];
                rb[i*N+j] = acc + bib[j];
            }
        }
    }

    /* Poison output */
    for (int i = 0; i < BATCH*M*N; i++) C[i] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, bias, C, M, N, K); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < BATCH*M*N; i++) {
        if (C[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    }
    hvx_report(errors, BATCH*M*N, fb,
               fb >= 0 ? (long)C[fb] : 0,
               fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
