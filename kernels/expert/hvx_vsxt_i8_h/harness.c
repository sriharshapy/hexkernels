#include "harness_common.h"
#include "kernel_api.h"

#define N 256        /* 2 full 128B input blocks, no tail */
#define NOUT N

static int8_t a[N] HVX_ALIGN;
static int16_t out[NOUT] HVX_ALIGN, ref[NOUT] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x5EC70u;
    for (int i = 0; i < N; i++)
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Pinned known values in block 0: identity ramp incl. negatives for
       hand-verification of sign-extension. */
    for (int i = 0; i < 128; i++) a[i] = (int8_t)(i - 64);  /* -64..63 */

    for (int base = 0; base + 128 <= N; base += 128)
        for (int j = 0; j < 64; j++) {
            ref[base + j]      = (int16_t)a[base + 2*j];
            ref[base + 64 + j] = (int16_t)a[base + 2*j + 1];
        }

    for (int i = 0; i < NOUT; i++) out[i] = (int16_t)0xA5A5;

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
