#include "harness_common.h"
#include "kernel_api.h"
#define N 1000
static int8_t  a[N]   HVX_ALIGN;
static int32_t out[1] HVX_ALIGN;
static int32_t ref[1];
int main(void) {
    uint32_t s = 0xB3C4D5u;
    for (int i = 0; i < N; i++) a[i] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Inject edge values: max-magnitude squares and negative-in-tail.
       a[999]=-128 (tail, max square = 16384), a[0]=127 (max positive square = 16129). */
    a[0]   =  127;
    a[999] = -128;
    /* Compute golden reference: sum of squares. */
    ref[0] = 0;
    for (int i = 0; i < N; i++) ref[0] += (int32_t)a[i] * (int32_t)a[i];
    /* Poison output so a no-op is detected. */
    out[0] = (int32_t)0xA5A5A5A5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, N, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = (out[0] != ref[0]) ? 1 : 0;
    hvx_report(errors, N, errors ? 0 : -1, (long)out[0], (long)ref[0]);
    return errors ? 1 : 0;
}
