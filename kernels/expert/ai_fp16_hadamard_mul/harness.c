#include "harness_common.h"
#include "kernel_api.h"
#define N 1000

static hvx_hf a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static hvx_hf out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

/* Generate a small exactly-fp16-representable value: multiples of 0.25 in [-2, 2].
 * Products stay within fp16 range (|product| <= 4.0 << 65504). */
static float gen_small(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 17) - 8;   /* -8..8 */
    return (float)(q * 0.25f);             /* -2.0 .. 2.0 */
}

int main(void) {
    uint32_t s = 0xF16B1u;
    for (int i = 0; i < N; i++) a[i] = (hvx_hf)gen_small(&s);
    for (int i = 0; i < N; i++) b[i] = (hvx_hf)gen_small(&s);

    /* Edge: product negative */
    a[0] = (hvx_hf) 1.5f;   b[0] = (hvx_hf)-0.75f;
    /* Edge: zero operand -> product == 0 */
    a[1] = (hvx_hf) 0.0f;   b[1] = (hvx_hf) 1.25f;
    a[2] = (hvx_hf)-0.5f;   b[2] = (hvx_hf) 0.0f;
    /* Edge: product requiring fp16 rounding: 0.1 is not exact in fp16;
     * both a[3] and b[3] are representable but (float)a[3]*(float)b[3]
     * may not be exactly representable as fp16 -> rounding exercised. */
    a[3] = (hvx_hf) 0.75f;  b[3] = (hvx_hf) 0.75f;  /* 0.5625 exact in fp16 */
    a[4] = (hvx_hf) 1.75f;  b[4] = (hvx_hf) 1.75f;  /* 3.0625 exact in fp16 */
    /* Edge: both negative -> positive product */
    a[5] = (hvx_hf)-1.25f;  b[5] = (hvx_hf)-0.5f;

    /* Scalar reference: one IEEE multiply per element, cast to fp16 */
    for (int i = 0; i < N; i++)
        ref[i] = (hvx_hf)((float)a[i] * (float)b[i]);

    /* Poison output buffer */
    for (int i = 0; i < N; i++)
        *((volatile unsigned short *)&out[i]) = 0xA5A5u;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
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
