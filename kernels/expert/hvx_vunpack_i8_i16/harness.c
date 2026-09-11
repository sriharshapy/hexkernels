#include "harness_common.h"
#include "kernel_api.h"

#define N 1000   /* 7*128 + 104 tail */

static int8_t a[N] HVX_ALIGN;
static int16_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x5A9C42u;
    for (int i = 0; i < N; i++)
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Pinned edge cases. */
    a[0] = -128;  /* most negative byte -> -128 (no overflow, int16 covers it) */
    a[1] = 127;   /* most positive byte -> 127 */
    a[2] = -1;    /* -1 -> -1 (sign bits must all extend) */
    a[3] = 0;     /* 0 -> 0 */
    a[N-1] = -100; /* tail-path edge (index 999, inside the 104-elem tail) */

    for (int i = 0; i < N; i++)
        ref[i] = (int16_t)(int8_t)a[i];

    for (int i = 0; i < N; i++) {
        unsigned char *p = (unsigned char *)&out[i];
        p[0] = 0xA5; p[1] = 0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N); });
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
