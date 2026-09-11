#include "harness_common.h"
#include "kernel_api.h"

#define G 3
#define NIN (64 * G)     /* 192 int16 elements per input */
#define NOUT (128 * G)   /* 384 int16 elements output */

static int16_t a[NIN] HVX_ALIGN, b[NIN] HVX_ALIGN;
static int16_t out[NOUT] HVX_ALIGN, ref[NOUT] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xC0B1u;
    for (int i = 0; i < NIN; i++) {
        a[i] = (int16_t)(hvx_lcg(&s) >> 16);
        b[i] = (int16_t)(hvx_lcg(&s) >> 16);
    }

    /* Pinned known pattern in block 0: identity ramps for hand-verification. */
    for (int i = 0; i < 64; i++) { a[i] = (int16_t)(1000 + i); b[i] = (int16_t)(-1000 - i); }

    for (int k = 0; k < G; k++)
        for (int i = 0; i < 64; i++) {
            ref[k*128 + i]      = a[k*64 + i];
            ref[k*128 + 64 + i] = b[k*64 + i];
        }

    for (int i = 0; i < NOUT; i++) out[i] = (int16_t)0xA5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, G); });
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
