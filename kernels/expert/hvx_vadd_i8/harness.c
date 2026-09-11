#include "harness_common.h"
#include "kernel_api.h"

#define N 1037   /* 8*128 + 13 tail */

static int8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xADD18u;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        b[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned wraparound edge cases. */
    a[0] = 127;  b[0] = 127;   /* 127+127 = -2 (wrap) */
    a[1] = 100;  b[1] = 100;   /* 100+100 = -56 (wrap) */
    a[2] = -128; b[2] = -1;    /* -128-1 = 127 (wrap) */
    a[3] = 127;  b[3] = 1;     /* 127+1 = -128 (wrap) */
    /* Tail-path edge: last element (index N-1 = 1036, inside the 13-elem tail). */
    a[N-1] = 50; b[N-1] = 50;

    for (int i = 0; i < N; i++)
        ref[i] = (int8_t)((int)a[i] + (int)b[i]);

    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
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
