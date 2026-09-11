#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N 512

static float y[N] HVX_ALIGN, dy[N] HVX_ALIGN;
static float dx[N] HVX_ALIGN, ref[N] HVX_ALIGN;

/* Generate fp32 u in [-4, 4] in steps of 0.5 (exact in fp32).
 * y = sigmoid(u) = 1 / (1 + expf(-u)) which lies in (0, 1). */
static float gen_sigmoid_y(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 17) - 8;  /* -8..8 -> maps to -4.0..4.0 in steps of 0.5 */
    float u = (float)q * 0.5f;
    return 1.0f / (1.0f + expf(-u));
}

/* Generate fp32 upstream gradient in [-2, 2] in steps of 0.25. */
static float gen_grad(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 17) - 8;
    return (float)(q * 0.25f);
}

int main(void) {
    uint32_t s = 0xDEADBEEFu;
    for (int i = 0; i < N; i++) y[i]  = gen_sigmoid_y(&s);
    for (int i = 0; i < N; i++) dy[i] = gen_grad(&s);

    /* ---- Pinned edge cases ---- */
    /* Edge 1: y=0.5 -> maximum gradient y*(1-y)=0.25 */
    y[0]  = 0.5f;    dy[0]  =  1.0f;   /* dx = 0.25 */
    /* Edge 2: y near 0 -> gradient near 0 */
    y[1]  = 0.0183f; dy[1]  =  1.0f;   /* sigmoid(-4): dx ~ 0.0179 */
    /* Edge 3: y near 1 -> gradient near 0 */
    y[2]  = 0.9817f; dy[2]  =  1.0f;   /* sigmoid(4): dx ~ 0.0179 */
    /* Edge 4: negative upstream gradient */
    y[3]  = 0.5f;    dy[3]  = -2.0f;   /* dx = -0.5 */
    /* Edge 5: y*dy with large magnitude */
    y[4]  = 0.7311f; dy[4]  =  2.0f;   /* sigmoid(1): dx ~ 0.393 */
    /* Edge 6: last element */
    y[N-1] = 0.5f;   dy[N-1] = 0.75f;

    /* Scalar reference: sigmoid backward */
    for (int i = 0; i < N; i++)
        ref[i] = dy[i] * y[i] * (1.0f - y[i]);

    /* Poison output buffer */
    for (int i = 0; i < N; i++) {
        uint32_t poison = 0xA5A5A5A5u;
        __builtin_memcpy(&dx[i], &poison, 4);
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(y, dy, dx, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    /* Tolerance compare (elementwise) */
    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++) {
        if (!hvx_close_f32(dx[i], ref[i])) {
            errors++;
            if (fb < 0) {
                fb = i;
                uint32_t g, e;
                __builtin_memcpy(&g, &dx[i],  4);
                __builtin_memcpy(&e, &ref[i], 4);
                gotv = (long)g; expv = (long)e;
            }
        }
    }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
