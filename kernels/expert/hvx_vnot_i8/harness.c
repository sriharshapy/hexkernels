#include "harness_common.h"
#include "kernel_api.h"

#define N 1017   /* 7*128 + 121 tail */

static int8_t a[N] HVX_ALIGN;
static int8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x907Eu;
    for (int i = 0; i < N; i++)
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);

    a[0] = 0;
    a[1] = -1;
    a[2] = (int8_t)0x0F;
    a[3] = 127;
    a[N-1] = (int8_t)0xAA;

    for (int i = 0; i < N; i++)
        ref[i] = (int8_t)(~a[i]);

    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

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
