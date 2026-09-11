#include "harness_common.h"
#include "kernel_api.h"

#define N 384        /* 3 full 128B input blocks, no tail */
#define NOUT N       /* widen preserves element count (doubles byte width, not count) */

static uint8_t a[N] HVX_ALIGN;
static uint16_t out[NOUT] HVX_ALIGN, ref[NOUT] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x2E70u;
    for (int i = 0; i < N; i++)
        a[i] = (uint8_t)(hvx_lcg(&s) >> 24);

    /* Pinned known pattern in block 0: identity ramp for hand-verification. */
    for (int i = 0; i < 128; i++) a[i] = (uint8_t)i;

    for (int base = 0; base + 128 <= N; base += 128)
        for (int j = 0; j < 64; j++) {
            ref[base + j]      = (uint16_t)a[base + 2*j];
            ref[base + 64 + j] = (uint16_t)a[base + 2*j + 1];
        }

    for (int i = 0; i < NOUT; i++) out[i] = 0xA5A5u;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < NOUT; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) fb = i;
        }
    }
    hvx_report(errors, NOUT, fb,
               fb >= 0 ? (long)out[fb] : 0L,
               fb >= 0 ? (long)ref[fb] : 0L);
    return errors ? 1 : 0;
}
