#include "harness_common.h"
#include "kernel_api.h"

#define N 577   /* 9*64 + 1 tail (64 int16 lanes/vector) */

static int16_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int16_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x191Bu;
    for (int i = 0; i < N; i++) {
        a[i] = (int16_t)(hvx_lcg(&s) >> 16);
        b[i] = (int16_t)(hvx_lcg(&s) >> 16);
    }

    /* Pinned edge cases: low-16-bit truncation, not a Q15 fixed-point multiply. */
    a[0] = 300;     b[0] = 300;     /* 90000 -> low16 = 24464 */
    a[1] = -300;    b[1] = 300;     /* -90000 -> low16 = -24464 */
    a[2] = 32767;   b[2] = 2;       /* 65534 -> low16 = -2 */
    a[3] = -32768;  b[3] = -1;      /* 32768 -> low16 = -32768 */
    /* Tail-path edge: last element (index N-1 = 576, the single-element tail). */
    a[N-1] = 7; b[N-1] = 9;         /* 63 (no truncation needed) */

    for (int i = 0; i < N; i++)
        ref[i] = (int16_t)((int)a[i] * (int)b[i]);

    for (int i = 0; i < N; i++) {
        unsigned char *p = (unsigned char *)&out[i];
        p[0] = 0xA5; p[1] = 0xA5;
    }

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
