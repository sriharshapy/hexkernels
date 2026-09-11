#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N 1024

static float x_buf[N] HVX_ALIGN;
static float out[N]   HVX_ALIGN;
static float ref[N]   HVX_ALIGN;

/* Pinned tanh activation.  ALL computation in fp32; output is fp32 (no cast).
 * Semantics: tanh_f32(x) = tanhf(x)  (Hexagon libm).
 * #include <math.h> required.  Do NOT change the form or constant types. */
static float tanh_f32(float x) {
    return tanhf(x);
}

int main(void) {
    /* Seed: eighth-integers in [-6, 6] (97 distinct values).
     * LCG picks from [-48, 48] mapped to q * 0.125f so tanh is non-trivial
     * on both negative and positive inputs. */
    uint32_t s = 0x511FE01u;
    for (int i = 0; i < N; i++) {
        int q = (int)(hvx_lcg(&s) % 97) - 48;   /* -48..48 -> x in [-6.0, 6.0] */
        x_buf[i] = (float)(q * 0.125f);
    }
    /* Edge cases pinned at fixed indices */
    x_buf[0] = 0.0f;     /* tanh(0) = 0 exactly */
    x_buf[1] = -6.0f;    /* large negative: tanh -> ~-1 */
    x_buf[2] = 6.0f;     /* large positive: tanh -> ~+1 */
    x_buf[3] = -0.125f;  /* small negative */
    x_buf[4] = 0.125f;   /* small positive */
    x_buf[5] = -3.0f;    /* mid-range negative */

    /* Reference: each element through the pinned tanh_f32 (same fn, same tanhf). */
    for (int i = 0; i < N; i++)
        ref[i] = tanh_f32(x_buf[i]);

    /* Poison output buffer: 0xA5A5A5A5 per fp32 element */
    for (int i = 0; i < N; i++) {
        uint32_t poison = 0xA5A5A5A5u;
        __builtin_memcpy(&out[i], &poison, 4);
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x_buf, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    /* Bit-exact compare: reinterpret float as uint32 and compare bit patterns. */
    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++) {
        uint32_t g, e;
        __builtin_memcpy(&g, &out[i], 4);
        __builtin_memcpy(&e, &ref[i], 4);
        if (!hvx_close_f32(out[i], ref[i])) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
