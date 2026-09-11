#include "harness_common.h"
#include "kernel_api.h"
#define M 32
#define N 32
#define K 66
static int16_t A[M*K]  HVX_ALIGN;
static int16_t B[K*N]  HVX_ALIGN;
static int32_t C[M*N]  HVX_ALIGN;
static int32_t ref[M*N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x4B9D2Eu;
    /* Bound inputs to |x| <= 127 so max |acc| = 127*127*130 = 2,097,290 << INT32_MAX */
    for (int i = 0; i < M*K; i++) A[i] = (int16_t)((int8_t)(hvx_lcg(&s) >> 24));
    for (int i = 0; i < K*N; i++) B[i] = (int16_t)((int8_t)(hvx_lcg(&s) >> 24));
    /* Edge cases: max positive product, sign in reduction */
    A[0] = 127;  B[0] = 127;        /* large positive */
    A[1] = -128; B[N] = 127;        /* sign in K-reduction */
    A[2] = 0;    B[2*N] = 127;      /* zero term */

    /* Golden reference */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
            ref[i*N+j] = acc;
        }
    }

    /* Poison */
    for (int i = 0; i < M*N; i++) C[i] = (int32_t)0xA5A5A5A5;

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
