#include "harness_common.h"
#include "kernel_api.h"

#define N 896   /* 7 full 128B blocks, no tail */

static int8_t a[N] HVX_ALIGN;
static int8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xDE17Au;
    for (int i = 0; i < N; i++)
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Pinned known pattern in block 0: identity ramp for hand-verification
       (a[i]=i for i in [0,128) -> reversed block = 127,126,...,0). */
    for (int i = 0; i < 128; i++) a[i] = (int8_t)i;

    for (int base = 0; base + 128 <= N; base += 128)
        for (int i = 0; i < 128; i++)
            ref[base + i] = a[base + 127 - i];

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
