#include "harness_common.h"
#include "kernel_api.h"
#define N 1024

static int32_t acc[N] HVX_ALIGN;
static int8_t  out[N] HVX_ALIGN;
static int8_t  ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x60D5u;
    /* acc in -2000..2000 -> r = (acc*17+8)>>4 in ~[-2125, 2125], so saturation to
     * [-128,127] fires for most elements. Seed a few exact boundary accumulators. */
    for (int i = 0; i < N; i++) acc[i] = (int32_t)((int)(hvx_lcg(&s) % 4001) - 2000);
    acc[0] = 120;    /* r = 128 -> saturates to 127 (just over) */
    acc[1] = 119;    /* r = 127 -> exact max, no saturation */
    acc[2] = -121;   /* r = -128 -> exact min */
    acc[3] = -122;   /* r = -129 -> saturates to -128 (just under) */
    acc[4] = 0;      /* r = 0 */

    for (int i = 0; i < N; i++) {
        int r = (acc[i] * 17 + 8) >> 4;
        if (r > 127) r = 127; else if (r < -128) r = -128;
        ref[i] = (int8_t)r;
    }

    for (int i = 0; i < N; i++) *((volatile signed char *)&out[i]) = (signed char)0x5A; /* poison */

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(acc, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
