#include "harness_common.h"
#include "kernel_api.h"
#include <string.h>

#define T_DIM   4
#define L_DIM   256
#define N       (T_DIM * L_DIM)   /* 1024 total elements */
#define ALPHA   0.5f

/* Generate a small exactly-fp32-representable value: quarter-integers in [-4, 4]. */
static float gen_f32(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 33) - 16;   /* -16..16 -> -4.0..4.0 in steps of 0.25 */
    return (float)(q * 0.25f);
}

static float x_buf[N]    HVX_ALIGN;   /* read-only input x */
static float y_orig[N]   HVX_ALIGN;   /* original y, preserved for reference */
static float y_work[N]   HVX_ALIGN;   /* copy of y passed to candidate (mutated in-place) */
static float ref[N]      HVX_ALIGN;   /* reference output */

int main(void) {
    uint32_t s = 0xFEEDFACEu;
    for (int i = 0; i < N; i++) x_buf[i]  = gen_f32(&s);
    for (int i = 0; i < N; i++) y_orig[i] = gen_f32(&s);

    /* Pinned edge cases. */
    /* Edge 0: y is negative; depends on magnitudes whether result flips sign */
    x_buf[0]  =  1.0f;  y_orig[0]  = -3.0f;   /* 0.5*1 + (-3) = -2.5 */
    /* Edge 1: x = 0, result == y (alpha term vanishes) */
    x_buf[1]  =  0.0f;  y_orig[1]  =  2.5f;   /* 0.5*0 + 2.5 = 2.5 */
    /* Edge 2: y = 0, result == alpha*x */
    x_buf[2]  =  3.0f;  y_orig[2]  =  0.0f;   /* 0.5*3 + 0 = 1.5 */
    /* Edge 3: rounding -- alpha*1e8 + 1e-8 (1e-8 is below ulp of 5e7) */
    x_buf[3]  =  1e8f;  y_orig[3]  =  1e-8f;  /* 0.5*1e8 + 1e-8 ~= 5e7 */
    /* Last element of last tensor */
    x_buf[N-1]  =  1.0f; y_orig[N-1]  =  1.0f; /* 0.5*1 + 1 = 1.5 */

    /* Compute reference: ref[i] = ALPHA * x[i] + y_orig[i] */
    for (int i = 0; i < N; i++)
        ref[i] = ALPHA * x_buf[i] + y_orig[i];

    /* Copy original y into work buffer; poison it first so a no-op fails. */
    for (int i = 0; i < N; i++) {
        uint32_t poison = 0xA5A5A5A5u;
        __builtin_memcpy(&y_work[i], &poison, 4);
    }
    memcpy(y_work, y_orig, sizeof(y_orig));

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x_buf, y_work, T_DIM, L_DIM, ALPHA); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) {
        if (!hvx_close_f32(y_work[i], ref[i])) {
            errors++;
            if (fb < 0) {
                fb = i;
            }
        }
    }
    /* Report first-bad as raw uint32 bit patterns widened to long. */
    long gotv = 0, expv = 0;
    if (fb >= 0) {
        uint32_t g, e;
        __builtin_memcpy(&g, &y_work[fb], 4);
        __builtin_memcpy(&e, &ref[fb],    4);
        gotv = (long)g; expv = (long)e;
    }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
