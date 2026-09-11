#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N 512

static float pred[N] HVX_ALIGN, tgt[N] HVX_ALIGN;
static float out[1] HVX_ALIGN, ref[1];

/* Generate fp32 in [-2, 2] in steps of 0.25 (quarter-integers, exact in fp32). */
static float gen_f32(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 17) - 8;  /* -8..8 -> -2.0..2.0 */
    return (float)(q * 0.25f);
}

int main(void) {
    uint32_t s = 0xDEADBEEFu;
    for (int i = 0; i < N; i++) pred[i] = gen_f32(&s);
    for (int i = 0; i < N; i++) tgt[i]  = gen_f32(&s);

    /* ---- Pinned edge cases ---- */
    /* Edge 1: zero difference (contributes 0 to MSE) */
    pred[0] = 1.0f;  tgt[0] = 1.0f;
    /* Edge 2: negative difference (squared -> still positive) */
    pred[1] = -1.5f; tgt[1] =  0.5f;  /* diff = -2.0, sq = 4.0 */
    /* Edge 3: large magnitude difference */
    pred[2] =  2.0f; tgt[2] = -2.0f;  /* diff = 4.0, sq = 16.0 */
    /* Edge 4: small difference */
    pred[3] =  0.25f; tgt[3] = 0.5f;  /* diff = -0.25, sq = 0.0625 */
    /* Edge 5: last element */
    pred[N-1] = 1.0f; tgt[N-1] = 0.0f; /* diff = 1.0, sq = 1.0 */

    /* Scalar reference: MSE = (1/n) * sum( (pred-tgt)^2 ) */
    float acc = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = pred[i] - tgt[i];
        acc += d * d;
    }
    ref[0] = acc / (float)N;

    /* Poison output */
    uint32_t poison = 0xA5A5A5A5u;
    __builtin_memcpy(&out[0], &poison, 4);

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(pred, tgt, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    /* Tolerance compare: fp32 reductions may reorder */
    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    if (!hvx_close_f32(out[0], ref[0])) {
        errors = 1; fb = 0;
        uint32_t g, e;
        __builtin_memcpy(&g, &out[0], 4);
        __builtin_memcpy(&e, &ref[0], 4);
        gotv = (long)g; expv = (long)e;
    }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
