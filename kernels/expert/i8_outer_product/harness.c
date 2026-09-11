#include "harness_common.h"
#include "kernel_api.h"
#define M 128
#define N 128
static int8_t  a[M]    HVX_ALIGN;
static int8_t  b[N]    HVX_ALIGN;
static int32_t C[M*N]  HVX_ALIGN;
static int32_t ref[M*N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x7C4B9Au;
    for (int i = 0; i < M; i++) a[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int j = 0; j < N; j++) b[j] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Edge cases: sign combinations, extremes, zero */
    a[0] = 127;  b[0] = 127;    /* max positive */
    a[1] = -128; b[1] = 127;    /* mixed sign */
    a[2] = 127;  b[2] = -128;   /* mixed sign (other direction) */
    a[3] = 0;    b[3] = 127;    /* zero row */
    a[4] = 127;  b[4] = 0;      /* zero col */
    a[5] = -128; b[5] = -128;   /* max negative * max negative = positive */

    /* Golden reference */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            ref[i*N+j] = (int32_t)a[i] * (int32_t)b[j];

    /* Poison */
    for (int i = 0; i < M*N; i++) C[i] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, C, M, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < M*N; i++) {
        if (C[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    }
    hvx_report(errors, M*N, fb, fb >= 0 ? (long)C[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
