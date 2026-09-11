#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N 1024

static hvx_hf x_buf[N] HVX_ALIGN;
static hvx_hf out[N]   HVX_ALIGN;
static hvx_hf ref[N]   HVX_ALIGN;

/* Pinned GELU tanh approximation.  ALL intermediates in fp32; ONE fp16 round at end.
 * Constants: 0.7978845608 = sqrt(2/pi), 0.044715 = standard tanh-GELU coefficient.
 * Do NOT change the constant values or intermediate types. */
static float gelu_f32(float x) {
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

int main(void) {
    /* Seed: quarter-integers in [-4, 4] (17 values: -4,-3.75,...,3.75,4).
     * Uses LCG to pick from [-16,16] mapped to q*0.25f so all 33 quarter-steps
     * are reachable and both negative and positive GELU inputs are dense. */
    uint32_t s = 0xF16Cu;
    for (int i = 0; i < N; i++) {
        int q = (int)(hvx_lcg(&s) % 33) - 16;   /* -16..16 -> x in [-4.0, 4.0] */
        x_buf[i] = (hvx_hf)((float)q * 0.25f);
    }
    /* Edge cases pinned at fixed indices */
    x_buf[0] = (hvx_hf)0.0f;           /* gelu(0) = 0 */
    x_buf[1] = (hvx_hf)(-4.0f);        /* large negative: gelu -> ~0 but != 0 */
    x_buf[2] = (hvx_hf)4.0f;           /* large positive: gelu -> ~x */
    x_buf[3] = (hvx_hf)(-0.25f);       /* small negative: gelu > 0 (relu fails) */
    x_buf[4] = (hvx_hf)0.25f;          /* small positive */

    /* Reference: compute entirely in fp32, single fp16 round at end. */
    for (int i = 0; i < N; i++)
        ref[i] = (hvx_hf)gelu_f32((float)x_buf[i]);

    /* Poison output buffer */
    for (int i = 0; i < N; i++)
        *((volatile unsigned short *)&out[i]) = 0xA5A5u;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x_buf, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++) {
        unsigned short g = *(unsigned short *)&out[i];
        unsigned short e = *(unsigned short *)&ref[i];
        if (!hvx_close_f16bits(g, e)) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report_u16(errors, N, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
