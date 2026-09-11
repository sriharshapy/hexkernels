#include "harness_common.h"
#include "kernel_api.h"

#define N 583   /* 9*64 + 7 tail */
#define SHIFT 4

static int16_t a[N] HVX_ALIGN;
static int16_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x00A51Cu;
    for (int i = 0; i < N; i++)
        a[i] = (int16_t)(hvx_lcg(&s) >> 16);

    /* Pinned edge cases. */
    a[0] = 1;                 /* 1<<4 = 16 */
    a[1] = -1;                /* -1<<4 wraps: 0xFFFF<<4 = 0xFFF0 = -16 */
    a[2] = (int16_t)0x4000;   /* 16384<<4 = 0x40000 truncated 16b = 0 */
    a[3] = (int16_t)0x8000;   /* -32768<<4 truncated 16b = 0 */
    a[N-1] = (int16_t)0x0FFF; /* tail-path edge: 4095<<4 = 0xFFF0 = -16 */

    for (int i = 0; i < N; i++)
        ref[i] = (int16_t)((uint16_t)a[i] << SHIFT);

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
