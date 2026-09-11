#include "harness_common.h"
#include "kernel_api.h"

#define N 900   /* 7*128 + 4 tail */

static uint8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static uint8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xA746u;
    for (int i = 0; i < N; i++) {
        a[i] = (uint8_t)(hvx_lcg(&s) >> 24);
        b[i] = (uint8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned rounding edge cases: rnd((a+b+1)>>1) differs from floor((a+b)>>1)
     * exactly when a+b is odd. */
    a[0] = 3;   b[0] = 4;    /* sum 7 (odd): rnd=4, floor=3 */
    a[1] = 0;   b[1] = 255;  /* sum 255 (odd): rnd=128, floor=127 */
    a[2] = 255; b[2] = 255;  /* sum 510 (even): rnd=floor=255 */
    a[3] = 1;   b[3] = 2;    /* sum 3 (odd): rnd=2, floor=1 */
    /* Tail-path edge: last element (index N-1 = 899, inside the 4-elem tail). */
    a[N-1] = 9; b[N-1] = 10; /* sum 19 (odd): rnd=10, floor=9 */

    for (int i = 0; i < N; i++)
        ref[i] = (uint8_t)(((int)a[i] + (int)b[i] + 1) >> 1);

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
