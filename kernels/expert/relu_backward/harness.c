#include "harness_common.h"
#include "kernel_api.h"
#define N 512

static float x[N] HVX_ALIGN, dy[N] HVX_ALIGN;
static float dx[N] HVX_ALIGN, ref[N] HVX_ALIGN;

/* Generate fp32 in [-2, 2] in steps of 0.25 (quarter-integers, exact in fp32). */
static float gen_f32(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 17) - 8;  /* -8..8 -> -2.0..2.0 */
    return (float)(q * 0.25f);
}

int main(void) {
    uint32_t s = 0xDEADBEEFu;
    for (int i = 0; i < N; i++) x[i]  = gen_f32(&s);
    for (int i = 0; i < N; i++) dy[i] = gen_f32(&s);

    /* ---- Pinned edge cases ---- */
    /* Edge 1: x > 0, dy > 0 -> dx = dy */
    x[0]  =  1.0f;  dy[0]  =  0.75f;
    /* Edge 2: x < 0 -> gate closed, dx = 0 */
    x[1]  = -1.0f;  dy[1]  =  0.5f;
    /* Edge 3: x == 0 -> subgradient 0 by convention */
    x[2]  =  0.0f;  dy[2]  =  1.0f;
    /* Edge 4: x > 0, dy < 0 -> gradient passes through negative */
    x[3]  =  0.5f;  dy[3]  = -1.5f;
    /* Edge 5: x < 0, dy < 0 -> still gated to 0 */
    x[4]  = -0.25f; dy[4]  = -2.0f;
    /* Edge 6: last element */
    x[N-1] =  1.5f; dy[N-1] =  0.25f;

    /* Scalar reference: ReLU backward */
    for (int i = 0; i < N; i++)
        ref[i] = (x[i] > 0.0f) ? dy[i] : 0.0f;

    /* Poison output buffer */
    for (int i = 0; i < N; i++) {
        uint32_t poison = 0xA5A5A5A5u;
        __builtin_memcpy(&dx[i], &poison, 4);
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, dy, dx, N); });
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
