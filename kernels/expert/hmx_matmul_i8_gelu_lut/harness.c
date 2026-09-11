#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N 32

static uint8_t A[N*N]   HVX_ALIGN;
static int8_t  B[N*N]   HVX_ALIGN;
static int32_t lut[256] HVX_ALIGN;
static int32_t out[N*N] HVX_ALIGN;
static int32_t ref[N*N] HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

int main(void) {
    uint32_t s = 0x6E10u;
    /* Small ranges so the requant field stays in [-128,127] (LUT index in range):
     * A 0..3, B -1..1, K=32 -> |acc| <= 96 -> |r| <= 102. */
    for (int i = 0; i < N*N; i++) A[i] = (uint8_t)(hvx_lcg(&s) % 4);
    for (int i = 0; i < N*N; i++) B[i] = (int8_t)((int)(hvx_lcg(&s) % 3) - 1);

    /* GELU (tanh approx) lookup table over t = idx-128, rounded to int32. */
    for (int idx = 0; idx < 256; idx++) {
        double t = (double)(idx - 128);
        double g = 0.5 * t * (1.0 + tanh(0.7978845608 * (t + 0.044715 * t * t * t)));
        lut[idx] = (int32_t)llround(g);
    }

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int acc = 0;
            for (int k = 0; k < N; k++) acc += (int)A[i*N+k] * (int)B[k*N+j];
            int r = sx12((acc * 17 + 8) >> 4);
            ref[i*N+j] = lut[r + 128];
        }

    for (int i = 0; i < N*N; i++) *((volatile int32_t *)&out[i]) = 0x5A5A5A5A; /* poison */

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, lut, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N*N; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    hvx_report(errors, N*N, fb, gotv, expv);
    return errors ? 1 : 0;
}
