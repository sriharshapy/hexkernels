#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>

#define R 5
#define C 90   /* NOT a multiple of 64 fp16-lanes -- real tail path */

static hvx_hf x[R*C] HVX_ALIGN, out[R*C] HVX_ALIGN, ref[R*C] HVX_ALIGN;
static hvx_hf gamma_v[C] HVX_ALIGN;

static float gen_hf(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 13) - 6;   /* -6..6 */
    return (float)(q * 0.5f);              /* -3.0..3.0, exact in fp16 */
}

int main(void) {
    uint32_t s = 0x9A5Eu;

    /* gamma: deterministic, guaranteed negative AND positive values. */
    for (int c = 0; c < C; c++)
        gamma_v[c] = (hvx_hf)((float)(((c % 7) - 3)) * 0.5f);   /* [-1.5, 1.5] */

    /* Row 0: all-zero (ms=0, relies on +1e-3 epsilon -- must not produce inf/nan). */
    for (int c = 0; c < C; c++) x[0*C + c] = (hvx_hf)0.0f;

    /* Row 1: one dominant large-magnitude element, rest small. */
    for (int c = 0; c < C; c++) x[1*C + c] = (hvx_hf)gen_hf(&s);
    x[1*C + 0] = (hvx_hf)100.0f;

    /* Rows 2..4: general random data (also exercises the tail path). */
    for (int r = 2; r < R; r++)
        for (int c = 0; c < C; c++)
            x[r*C + c] = (hvx_hf)gen_hf(&s);
    x[2*C + 0] = (hvx_hf)3.0f;
    x[2*C + 1] = (hvx_hf)(-3.0f);
    x[(R-1)*C + (C-1)] = (hvx_hf)0.0f;   /* zero at the tail boundary */

    /* Independent scalar-float golden reference (same formula, duplicated
     * here -- never calls baseline.c/expert.c). */
    for (int r = 0; r < R; r++) {
        const hvx_hf *xr = x + (long)r * C;
        hvx_hf *refr = ref + (long)r * C;

        float sumsq = 0.0f;
        for (int c = 0; c < C; c++) { float v = (float)xr[c]; sumsq += v * v; }
        float ms = sumsq / (float)C;
        float inv_rms = 1.0f / sqrtf(ms + 1e-3f);

        for (int c = 0; c < C; c++)
            refr[c] = (hvx_hf)((float)xr[c] * inv_rms * (float)gamma_v[c]);
    }

    for (int i = 0; i < R*C; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, gamma_v, out, R, C); });
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
