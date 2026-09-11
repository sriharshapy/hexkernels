#include "harness_common.h"
#include "kernel_api.h"

#define N 777   /* 6*128 + 9 tail */

static uint8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static uint8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x5A7Bu;
    for (int i = 0; i < N; i++) {
        a[i] = (uint8_t)(hvx_lcg(&s) >> 24);
        b[i] = (uint8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned saturation edge cases. */
    a[0] = 200; b[0] = 200;   /* 400 -> sat 255 */
    a[1] = 255; b[1] = 1;     /* 256 -> sat 255 */
    a[2] = 0;   b[2] = 0;     /* 0 -> 0 (no sat) */
    a[3] = 100; b[3] = 50;    /* 150 -> 150 (no sat) */
    /* Tail-path edge: last element (index N-1 = 776, inside the 9-elem tail). */
    a[N-1] = 255; b[N-1] = 255;   /* 510 -> sat 255 */

    for (int i = 0; i < N; i++) {
        int r = (int)a[i] + (int)b[i];
        if (r > 255) r = 255;
        ref[i] = (uint8_t)r;
    }

    for (int i = 0; i < N; i++) out[i] = (uint8_t)0xA5;

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
