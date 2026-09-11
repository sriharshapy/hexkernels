#include "harness_common.h"
#include "kernel_api.h"
#define N 1024
static int8_t  a[N] HVX_ALIGN, w[N] HVX_ALIGN;
static int32_t out[1] HVX_ALIGN;
/* Overflow guard: max |w[i]*a[i]| = 128*128=16384; n*16384 = 1024*16384 = 16,777,216 < 2^31. */
int main(void) {
    uint32_t s = 0x7788AAu;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        w[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }
    /* Edge cases: max positive product, max negative product, zero weight, zero data. */
    a[0]  =  127; w[0]  =  127;  /* large positive */
    a[1]  =  127; w[1]  = -128;  /* large negative */
    a[2]  =    0; w[2]  =  127;  /* zero data -> zero contribution */
    a[3]  =  100; w[3]  =    0;  /* zero weight -> zero contribution */
    a[1023] = -128; w[1023] = -128; /* both negative -> positive product */
    /* Compute golden reference. */
    int32_t ref = 0;
    for (int i = 0; i < N; i++) ref += (int32_t)a[i] * (int32_t)w[i];
    /* Poison output. */
    out[0] = (int32_t)0xA5A5A5A5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, w, N, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = (out[0] != ref) ? 1 : 0;
    hvx_report(errors, N, errors ? 0 : -1, (long)out[0], (long)ref);
    return errors ? 1 : 0;
}
