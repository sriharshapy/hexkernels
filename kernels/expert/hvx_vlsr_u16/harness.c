#include "harness_common.h"
#include "kernel_api.h"

#define N 619   /* 9*64 + 43 tail */
#define SHIFT 5

static uint16_t a[N] HVX_ALIGN;
static uint16_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x00157Cu;
    for (int i = 0; i < N; i++)
        a[i] = (uint16_t)(hvx_lcg(&s) >> 16);

    /* Pinned edge cases. */
    a[0] = 1;        /* 1>>5 = 0 */
    a[1] = 0xFFFF;    /* 65535>>5 = 2047 */
    a[2] = 0x8000;    /* 32768>>5 = 1024 (no sign extension) */
    a[3] = 3;         /* 3>>5 = 0 */
    a[N-1] = 0x8421;  /* tail-path edge: 33825>>5 = 1057 */

    for (int i = 0; i < N; i++)
        ref[i] = (uint16_t)(a[i] >> SHIFT);

    for (int i = 0; i < N; i++) {
        unsigned char *p = (unsigned char *)&out[i];
        p[0] = 0xA5; p[1] = 0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N, SHIFT); });
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
