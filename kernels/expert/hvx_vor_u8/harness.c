#include "harness_common.h"
#include "kernel_api.h"

#define N 1033   /* 8*128 + 9 tail */

static uint8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static uint8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x081Bu;
    for (int i = 0; i < N; i++) {
        a[i] = (uint8_t)(hvx_lcg(&s) >> 24);
        b[i] = (uint8_t)(hvx_lcg(&s) >> 24);
    }

    a[0] = 0xF0; b[0] = 0x0F;
    a[1] = 0x00; b[1] = 0x00;
    a[2] = 0xAA; b[2] = 0x55;
    a[3] = 0xFF; b[3] = 0x00;
    a[N-1] = 0x30; b[N-1] = 0x0C;

    for (int i = 0; i < N; i++)
        ref[i] = (uint8_t)(a[i] | b[i]);

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
