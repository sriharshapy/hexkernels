#include "harness_common.h"
#include "kernel_api.h"
#define M 8
#define N 8
#define K 262   /* deep-K, 262 = 65*4 + 2 -- NOT a multiple of 4 (tail group) */

static int8_t  A[M*K] HVX_ALIGN;
static int8_t  B[K*N] HVX_ALIGN;
static int32_t C[M*N] HVX_ALIGN;
static int32_t ref[M*N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x7C3E19A1u;

    for (int i = 0; i < M*K; i++) A[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < K*N; i++) B[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    A[0] = 127;  B[0] = 127;              /* large positive contribution (i=0,k=0,j=0) */
    A[1] = -128; B[N] = 127;              /* sign-mismatch cancellation at k=1 */
    /* K tail: non-zero values at the last two k's (260, 261) for every row/col */
    for (int i = 0; i < M; i++) { A[i*K+260] = 100; A[i*K+261] = -50; }
    for (int j = 0; j < N; j++) { B[260*N+j] = -80; B[261*N+j] = 60; }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
            ref[i*N+j] = acc;
        }
    }

    for (int i = 0; i < M*N; i++) C[i] = (int32_t)0xA5A5A5A5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, C, M, N, K); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < M*N; i++) {
        if (C[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    }
    hvx_report(errors, M*N, fb, fb >= 0 ? (long)C[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
