#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#define N 2048   /* 32 HVX vectors of 64 fp16 lanes -- clean single-tile size */

static hvx_hf x[N] HVX_ALIGN, gamma_v[N] HVX_ALIGN, beta_v[N] HVX_ALIGN;
static hvx_hf out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

static float gen_hf(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 13) - 6;   /* -6..6 */
    return (float)(q * 0.5f);              /* -3.0..3.0, exact in fp16 */
}

int main(void) {
    uint32_t s = 0x1A9E5u;
    for (int i = 0; i < N; i++) x[i] = (hvx_hf)gen_hf(&s);

    /* Pinned per-feature affine params (runtime, must be read -- anti-hardcode):
     * gamma cycles 0.5,0.75,1.0,1.25,1.5 ; beta cycles -0.75..0.75 (nonzero for
     * most indices, catching an omitted-beta near-miss). Both exact in fp16. */
    for (int i = 0; i < N; i++) {
        gamma_v[i] = (hvx_hf)(1.0f + 0.25f * (float)((i % 5) - 2));
        beta_v[i]  = (hvx_hf)(0.25f * (float)((i % 7) - 3));
    }
    /* Edge cases: extreme values at the ends of the reduction range. */
    x[0] = (hvx_hf)3.0f; x[1] = (hvx_hf)(-3.0f);
    x[N-1] = (hvx_hf)0.0f;

    /* Independent scalar golden: float mean/var/inv_std, float affine, then
     * fp16-round -- cross-checked against baseline.c/expert.c both. */
    float sumx = 0.0f, sumsq = 0.0f;
    for (int i = 0; i < N; i++) {
        float v = (float)x[i];
        sumx += v; sumsq += v * v;
    }
    float mean = sumx / (float)N;
    float var  = sumsq / (float)N - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv_std = 1.0f / sqrtf(var + 1e-3f);
    for (int i = 0; i < N; i++) {
        float d = ((float)x[i] - mean) * inv_std;
        ref[i] = (hvx_hf)(d * (float)gamma_v[i] + (float)beta_v[i]);
    }

    for (int i = 0; i < N; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, gamma_v, beta_v, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++) {
        unsigned short g = *(unsigned short *)&out[i], e = *(unsigned short *)&ref[i];
        if (!hvx_close_f16bits(g, e)) {
            errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report_u16(errors, N, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
