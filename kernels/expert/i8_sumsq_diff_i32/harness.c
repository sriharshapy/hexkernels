#include "harness_common.h"
#include "kernel_api.h"
#define N 1024
static int8_t  a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int32_t out[1] HVX_ALIGN;
/* Overflow guard: max |a[i]-b[i]| = 255, so (a[i]-b[i])^2 <= 65025.
   n*65025 = 1024*65025 = 66,585,600 < 2^31, so int32 accumulator is safe. */
int main(void) {
    uint32_t s = 0x3C4D5Eu;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        b[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }
    /* Edge cases. */
    a[0]   =  127; b[0]   = -128; /* max positive diff: 255^2 = 65025 */
    a[1]   = -128; b[1]   =  127; /* max negative diff (sign symmetry) */
    a[2]   =    0; b[2]   =    0; /* zero diff */
    a[3]   =   64; b[3]   =   64; /* same positive value */
    a[1023] = -100; b[1023] = 100; /* tail: diff = -200, sq = 40000 */
    /* Compute golden reference. */
    int32_t ref = 0;
    for (int i = 0; i < N; i++) {
        int32_t d = (int32_t)a[i] - (int32_t)b[i];
        ref += d * d;
    }
    /* Poison output. */
    out[0] = (int32_t)0xA5A5A5A5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, N, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = (out[0] != ref) ? 1 : 0;
    hvx_report(errors, N, errors ? 0 : -1, (long)out[0], (long)ref);
    return errors ? 1 : 0;
}
