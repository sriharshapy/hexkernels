#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define R 8
#define C 256   /* 4 HVX vectors of 64 fp16 lanes per row -- clean single-tile size */

static hvx_hf x[R*C] HVX_ALIGN, out[R*C] HVX_ALIGN, ref[R*C] HVX_ALIGN;

static float gen_hf(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 13) - 6;   /* -6..6 */
    return (float)(q * 0.5f);              /* -3.0..3.0, exact in fp16 */
}

int main(void) {
    uint32_t s = 0x50F7u;
    for (int i = 0; i < R*C; i++) x[i] = (hvx_hf)gen_hf(&s);

    /* Edge cases: one row with a single dominant max, one row with all
     * identical values (uniform softmax = 1/C every element). */
    for (int j = 0; j < C; j++) x[0*C + j] = (hvx_hf)(-3.0f);
    x[0*C + 7] = (hvx_hf)3.0f;               /* single dominant max */
    for (int j = 0; j < C; j++) x[1*C + j] = (hvx_hf)1.0f;  /* uniform row */

    /* Independent scalar golden: float max-reduce, float exp, float sum,
     * float divide, then fp16-round -- cross-checked against
     * baseline.c/expert.c both. */
    for (int r = 0; r < R; r++) {
        const hvx_hf *xr = x + r*C;
        hvx_hf *refr = ref + r*C;
        float rowmax = (float)xr[0];
        for (int j = 1; j < C; j++) if ((float)xr[j] > rowmax) rowmax = (float)xr[j];
        float e[C];
        float rowsum = 0.0f;
        for (int j = 0; j < C; j++) { e[j] = expf((float)xr[j] - rowmax); rowsum += e[j]; }
        for (int j = 0; j < C; j++) refr[j] = (hvx_hf)(e[j] / rowsum);
    }

    for (int i = 0; i < R*C; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, R, C); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < R*C; i++) {
        unsigned short g = *(unsigned short *)&out[i], e = *(unsigned short *)&ref[i];
        if (!hvx_close_f16bits(g, e)) {
            errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report_u16(errors, R*C, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
