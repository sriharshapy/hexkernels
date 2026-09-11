#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N 1000

static hvx_hf x[N] HVX_ALIGN, out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xF16Bu;
    /* Seed with quarter-integers in [-4, 10] so both clamp bounds are exercised. */
    for (int i = 0; i < N; i++) {
        /* range: -4.0 .. +10.0 in steps of 0.25; period = 57 values */
        float v = -4.0f + (float)(hvx_lcg(&s) % 57) * 0.25f;
        x[i] = (hvx_hf)v;
    }
    /* Force specific edge cases at known indices. */
    x[0] = (hvx_hf)0.0f;    /* exactly lower bound */
    x[1] = (hvx_hf)6.0f;    /* exactly upper bound */
    x[2] = (hvx_hf)(-1.0f); /* below 0 -> clamp to 0 */
    x[3] = (hvx_hf)7.5f;    /* above 6 -> clamp to 6 */
    x[4] = (hvx_hf)(-0.25f);/* small negative */
    x[5] = (hvx_hf)3.0f;    /* mid-range, no clamp */

    /* Scalar reference: fminf(fmaxf(x,0),6), cast back to fp16. */
    for (int i = 0; i < N; i++) {
        float v = (float)x[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        ref[i] = (hvx_hf)v;
    }

    /* Poison output buffer (fp16 sentinel 0xA5A5). */
    for (int i = 0; i < N; i++)
        *((volatile unsigned short *)&out[i]) = 0xA5A5u;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++) {
        unsigned short g = *(unsigned short *)&out[i];
        unsigned short e = *(unsigned short *)&ref[i];
        if (g != e) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report_u16(errors, N, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
