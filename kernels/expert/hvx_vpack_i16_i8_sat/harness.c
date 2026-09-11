#include "harness_common.h"
#include "kernel_api.h"

#define G 2
#define NIN (64 * G)     /* 128 int16 elements per input */
#define NOUT (128 * G)   /* 256 int8 elements output */

static int16_t a[NIN] HVX_ALIGN, b[NIN] HVX_ALIGN;
static int8_t out[NOUT] HVX_ALIGN, ref[NOUT] HVX_ALIGN;

static int8_t sat8(int x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

int main(void) {
    uint32_t s = 0xACC7u;
    for (int i = 0; i < NIN; i++) {
        a[i] = (int16_t)((int)(hvx_lcg(&s) >> 16) % 300 - 150);
        b[i] = (int16_t)((int)(hvx_lcg(&s) >> 16) % 300 - 150);
    }

    /* Pinned saturation edge cases in block 0. */
    a[0] = 1000;   b[0] = -1000;   /* both saturate: a->127, b->-128 */
    a[1] = 100;    b[1] = -100;    /* no saturation needed */
    a[2] = 127;    b[2] = -128;    /* exact boundary, no saturation */
    a[3] = 128;    b[3] = -129;    /* just past boundary: saturate */

    for (int k = 0; k < G; k++)
        for (int i = 0; i < 64; i++) {
            ref[k*128 + i]      = sat8(b[k*64 + i]);
            ref[k*128 + 64 + i] = sat8(a[k*64 + i]);
        }

    for (int i = 0; i < NOUT; i++) out[i] = (int8_t)0xA5;

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
