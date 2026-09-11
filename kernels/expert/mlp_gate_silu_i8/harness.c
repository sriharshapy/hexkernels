#include "harness_common.h"
#include "kernel_api.h"

#define N 150   /* NOT a multiple of 128: one full vector + 22-element tail */

static int8_t gate[N]      HVX_ALIGN;
static int8_t up[N]        HVX_ALIGN;
static int8_t silu_lut[256] HVX_ALIGN;
static int8_t out[N]       HVX_ALIGN;
static int8_t ref[N]       HVX_ALIGN;

/* tanh via a rational Pade approximation (avoids a libm dependency in the
 * bare-metal sim -- same technique used elsewhere in this repo). */
static float tanh_approx(float t) {
    if (t >  4.0f) return  1.0f;
    if (t < -4.0f) return -1.0f;
    float x2 = t * t;
    return t * (27.0f + x2) / (27.0f + 9.0f * x2);
}

/* silu_lut[idx] = clamp(round(v/(1+exp(-v/16))), -128, 127), v = idx-128.
 * v/(1+exp(-v/16)) == v * sigmoid(v/16); sigmoid(x) = 0.5*(1+tanh(x/2)).
 * Two variants (scale on v) to prevent hardcoding. */
static void build_silu_lut(int8_t *lut, int variant) {
    for (int i = 0; i < 256; i++) {
        float v  = (float)(i - 128);
        float vv = (variant == 1) ? v * 0.8f : v;
        float x  = vv / 16.0f;
        float th = tanh_approx(x * 0.5f);
        float sig = 0.5f * (1.0f + th);
        float g  = vv * sig;
        int r = (int)(g + (g >= 0 ? 0.5f : -0.5f));
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        lut[i] = (int8_t)r;
    }
}

static int8_t requant_ref(int32_t prod, int32_t mult, int shift) {
    int64_t r    = (int64_t)prod * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q    = (r >= 0) ? ((r + half) >> shift) : -(((-r) + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static const int32_t SMULTS[]  = { 3, 5 };
static const int     SSHIFTS[] = { 7, 3 };
#define NSETS ((int)(sizeof(SMULTS)/sizeof(SMULTS[0])))

int main(void) {
    uint32_t s = 0x9A31D07u;
    for (int i = 0; i < N; i++) gate[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < N; i++) up[i]   = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases: LUT index boundaries */
    gate[0] = -128;  /* idx = 0 */
    gate[1] =  127;  /* idx = 255 */
    gate[2] =  0;    /* idx = 128 */
    up[0] = 127; up[1] = -128; up[2] = 100;

    /* Tail: N=150 -> body[0..127], tail[128..149] (22 elements). Force
     * nonzero, sign-varied values in the tail. */
    for (int i = 128; i < N; i++) {
        gate[i] = (int8_t)(20 - 2*(i - 128));   /* varies across pos/neg */
        up[i]   = (int8_t)(-10 + 3*(i - 128));
    }

    int errors = 0, fb = -1; long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        build_silu_lut(silu_lut, p);
        int32_t mult = SMULTS[p];
        int     shift = SSHIFTS[p];

        for (int i = 0; i < N; i++) {
            uint8_t idx = (uint8_t)((int)gate[i] + 128);
            int32_t g = silu_lut[idx];
            int32_t prod = g * (int32_t)up[i];
            ref[i] = requant_ref(prod, mult, shift);
        }

        for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(gate, up, silu_lut, out, N, mult, shift); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < N; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = p*N + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
        }
    }
    hvx_report(errors, N*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
