#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N 1000

static float x[N] HVX_ALIGN, out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xF32Bu;
    /* Seed with eighth-integers in [-4, 10] so both clamp bounds are exercised. */
    for (int i = 0; i < N; i++) {
        /* range: -4.0 .. +10.0 in steps of 0.125; period = 113 values */
        float v = -4.0f + (float)(hvx_lcg(&s) % 113) * 0.125f;
        x[i] = v;
    }
    /* Force specific edge cases at known indices. */
    x[0] = 0.0f;    /* exactly lower bound */
    x[1] = 6.0f;    /* exactly upper bound */
    x[2] = -1.0f;   /* below 0 -> clamp to 0 */
    x[3] = 7.5f;    /* above 6 -> clamp to 6 */
    x[4] = -0.125f; /* small negative */
    x[5] = 3.0f;    /* mid-range, no clamp */

    /* Scalar reference: fminf(fmaxf(x,0),6) -- uniquely defined, bit-exact. */
    for (int i = 0; i < N; i++) {
        float v = x[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        ref[i] = v;
    }

    /* Poison output buffer: 0xA5A5A5A5 per fp32 element (4 bytes each). */
    for (int i = 0; i < N; i++) {
        uint32_t poison = 0xA5A5A5A5u;
        __builtin_memcpy(&out[i], &poison, 4);
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    /* Bit-exact compare: reinterpret each float as uint32 and compare bit patterns.
     * min/max against exact fp32 constants is uniquely determined -> scalar == HVX. */
    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++) {
        uint32_t g, e;
        __builtin_memcpy(&g, &out[i], 4);
        __builtin_memcpy(&e, &ref[i], 4);
        if (g != e) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
