#include "harness_common.h"
#include "kernel_api.h"

#define N 1013   /* 7*128 + 117 tail */

static int8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN, c[N] HVX_ALIGN;
static int8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x30E4u;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        b[i] = (int8_t)(hvx_lcg(&s) >> 24);
        c[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned edge cases. */
    a[0]=10; b[0]=5;  c[0]=99;   /* a>b true: out=a=10 (not c) */
    a[1]=5;  b[1]=10; c[1]=99;   /* a>b false: out=c=99 (not b) */
    a[2]=5;  b[2]=5;  c[2]=42;   /* equal, strict >: false: out=c=42 */
    a[3]=127; b[3]=-128; c[3]=0; /* extremes true: out=a=127 */
    /* Tail-path edge: last element (index N-1 = 1012, inside the 117-elem tail). */
    a[N-1]=-1; b[N-1]=-2; c[N-1]=77; /* a>b true: out=a=-1 */

    for (int i = 0; i < N; i++)
        ref[i] = (a[i] > b[i]) ? a[i] : c[i];

    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, c, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) fb = i;
        }
    }
    hvx_report(errors, N, fb,
               fb >= 0 ? (long)out[fb] : 0L,
               fb >= 0 ? (long)ref[fb] : 0L);
    return errors ? 1 : 0;
}
