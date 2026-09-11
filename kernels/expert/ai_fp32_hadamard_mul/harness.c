#include "harness_common.h"
#include "kernel_api.h"
#include <string.h>
#define N 1000

static float a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static float out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

/* Generate a small exactly-fp32-representable value: multiples of 0.125 in [-2, 2].
 * Products stay well within fp32 range (|product| <= 4.0 << 3.4e38). */
static float gen_small(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 33) - 16;  /* -16..16 */
    return (float)(q * 0.125f);            /* -2.0 .. 2.0 in steps of 0.125 */
}

int main(void) {
    uint32_t s = 0xF32B1u;
    for (int i = 0; i < N; i++) a[i] = gen_small(&s);
    for (int i = 0; i < N; i++) b[i] = gen_small(&s);

    /* Edge: product negative */
    a[0] =  1.5f;    b[0] = -0.75f;
    /* Edge: zero operand -> product == 0 */
    a[1] =  0.0f;    b[1] =  1.25f;
    a[2] = -0.5f;    b[2] =  0.0f;
    /* Edge: product requiring fp32 rounding:
     * 1.0/3.0 is not exactly representable; multiply by 3.0 exercises rounding. */
    a[3] =  0.333333343f;  b[3] =  3.0f;   /* ~1.0 but rounded */
    /* Edge: both negative -> positive product */
    a[4] = -1.25f;   b[4] = -0.5f;
    /* Edge: result near representability boundary */
    a[5] =  1.000000119f;  b[5] =  1.000000119f;  /* (1 + 2^-23)^2 exercises round-to-even */

    /* Scalar reference: one IEEE multiply per element */
    for (int i = 0; i < N; i++)
        ref[i] = a[i] * b[i];

    /* Poison output buffer: write 0xA5 into every byte */
    {
        unsigned char *p = (unsigned char *)out;
        for (int i = 0; i < (int)(N * sizeof(float)); i++) p[i] = 0xA5u;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++) {
        uint32_t g, e;
        memcpy(&g, &out[i], 4);
        memcpy(&e, &ref[i], 4);
        if (!hvx_close_f32(out[i], ref[i])) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
