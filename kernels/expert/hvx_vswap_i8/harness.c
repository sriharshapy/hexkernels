#include "harness_common.h"
#include "kernel_api.h"

#define N 1005   /* 7*128 + 109 tail */

static int8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t out_hi[N] HVX_ALIGN, out_lo[N] HVX_ALIGN;
static int8_t ref_hi[N], ref_lo[N];

int main(void) {
    uint32_t s = 0x5AAF1u;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        b[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned edge cases. */
    a[0] = 100; b[0] = -50;   /* pred true: hi=100, lo=-50 */
    a[1] = -50; b[1] = 100;   /* pred false: hi=100, lo=-50 (symmetric check) */
    a[2] = 5;   b[2] = 5;     /* equal -> pred false (strict >): hi=5, lo=5 */
    a[3] = 127; b[3] = -128;  /* extremes: pred true: hi=127, lo=-128 */
    /* Tail-path edge: last element (index N-1 = 1004, inside the 109-elem tail). */
    a[N-1] = -1; b[N-1] = 1;

    for (int i = 0; i < N; i++) {
        int pred = a[i] > b[i];
        ref_hi[i] = pred ? a[i] : b[i];
        ref_lo[i] = pred ? b[i] : a[i];
    }

    for (int i = 0; i < N; i++) { out_hi[i] = (int8_t)0xA5; out_lo[i] = (int8_t)0xA5; }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out_hi, out_lo, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long got = 0, exp = 0;
    for (int i = 0; i < N; i++) {
        if (out_hi[i] != ref_hi[i]) {
            errors++;
            if (fb < 0) { fb = i; got = out_hi[i]; exp = ref_hi[i]; }
        }
        if (out_lo[i] != ref_lo[i]) {
            errors++;
            if (fb < 0) { fb = i; got = out_lo[i]; exp = ref_lo[i]; }
        }
    }
    hvx_report(errors, N, fb, got, exp);
    return errors ? 1 : 0;
}
