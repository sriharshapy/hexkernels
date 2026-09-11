#include "harness_common.h"
#include "kernel_api.h"
#define M 256
#define K 130
static int8_t  A[M*K] HVX_ALIGN;
static int8_t  x[K]   HVX_ALIGN;
static int32_t y[M]   HVX_ALIGN;
static int32_t ref[M] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xD4E8A2u;
    for (int i = 0; i < M*K; i++) A[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int k = 0; k < K; k++) x[k] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Edge cases: large products, sign in reduction, K%128 tail */
    A[0] = 127; x[0] = 127;       /* large positive */
    A[K] = -128; x[1] = 127;      /* sign edge in row 1 */

    /* Golden reference */
    for (int i = 0; i < M; i++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++)
            acc += (int32_t)A[i*K+k] * (int32_t)x[k];
        ref[i] = acc;
    }

    /* Poison output */
    for (int i = 0; i < M; i++) y[i] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, x, y, M, K); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < M; i++) {
        if (y[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    }
    hvx_report(errors, M, fb, fb >= 0 ? (long)y[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
