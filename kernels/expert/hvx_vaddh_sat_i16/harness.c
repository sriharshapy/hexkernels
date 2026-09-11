#include "harness_common.h"
#include "kernel_api.h"

#define N 703   /* 10*64 + 63 tail */

static int16_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int16_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

static int16_t sat16(int v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

int main(void) {
    uint32_t s = 0x00ADD5u;
    for (int i = 0; i < N; i++) {
        a[i] = (int16_t)(hvx_lcg(&s) >> 16);
        b[i] = (int16_t)(hvx_lcg(&s) >> 16);
    }

    a[0] = 30000; b[0] = 10000;   /* saturates to 32767 */
    a[1] = -30000; b[1] = -10000; /* saturates to -32768 */
    a[2] = 100; b[2] = 200;       /* no-sat -> 300 */
    a[3] = -100; b[3] = -200;     /* no-sat -> -300 */
    a[N-1] = 32767; b[N-1] = 1;   /* tail-path edge: saturates to 32767 */

    for (int i = 0; i < N; i++)
        ref[i] = sat16((int)a[i] + (int)b[i]);

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
