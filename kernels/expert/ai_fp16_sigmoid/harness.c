#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N 1024

static hvx_hf x_buf[N] HVX_ALIGN;
static hvx_hf out[N]   HVX_ALIGN;
static hvx_hf ref[N]   HVX_ALIGN;

/* Pinned sigmoid.  ALL intermediates in fp32; ONE fp16 round at end.
 * Definition: sigmoid(x) = 1.0f / (1.0f + expf(-x))
 * Do NOT change the formula or intermediate types. */
static float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int main(void) {
    /* Seed: quarter-integers in [-6, 6] (49 values: -6,-5.75,...,5.75,6).
     * Uses LCG to pick from [-24,24] mapped to q*0.25f so all 49 quarter-steps
     * are reachable and both negative and positive sigmoid inputs are dense. */
    uint32_t s = 0x516Du;
    for (int i = 0; i < N; i++) {
        int q = (int)(hvx_lcg(&s) % 49) - 24;   /* -24..24 -> x in [-6.0, 6.0] */
        x_buf[i] = (hvx_hf)((float)q * 0.25f);
    }
    /* Edge cases pinned at fixed indices */
    x_buf[0] = (hvx_hf)0.0f;           /* sigmoid(0) = 0.5 */
    x_buf[1] = (hvx_hf)(-6.0f);        /* large negative: sigmoid -> ~0 but != 0 */
    x_buf[2] = (hvx_hf)6.0f;           /* large positive: sigmoid -> ~1 but != 1 */
    x_buf[3] = (hvx_hf)(-0.25f);       /* small negative: sigmoid < 0.5 (relu fails) */
    x_buf[4] = (hvx_hf)0.25f;          /* small positive: sigmoid > 0.5 */

    /* Reference: compute entirely in fp32, single fp16 round at end. */
    for (int i = 0; i < N; i++)
        ref[i] = (hvx_hf)sigmoid_f32((float)x_buf[i]);

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
