#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#include <stdint.h>

#define N 1000   /* 7*128 + 104 -- tail path */
#define S 0.05

static int8_t x[N]     HVX_ALIGN;
static int8_t out[N]   HVX_ALIGN;
static int8_t ref[N]   HVX_ALIGN;
static int8_t lut[256] HVX_ALIGN;

static int clamp_i8(double v) {
    if (v > 127.0) return 127;
    if (v < -128.0) return -128;
    return (int)v;
}

/* Real quantized SiLU table, built once at runtime (never hardcoded inside
 * the candidate): dequantize the index as a signed int8 value * S, compute
 * SiLU = x/(1+exp(-x)) in double precision, requantize by the same S,
 * round-to-nearest, clamp to int8. */
static void build_silu_lut(int8_t *L) {
    for (int k = 0; k < 256; k++) {
        int sv = (k >= 128) ? (k - 256) : k;
        double xf = (double)sv * S;
        double g = xf / (1.0 + exp(-xf));
        double q = g / S;
        double rounded = (q >= 0.0) ? floor(q + 0.5) : ceil(q - 0.5);
        L[k] = (int8_t)clamp_i8(rounded);
    }
}

int main(void) {
    uint32_t s = 0x51105113u;

    build_silu_lut(lut);

    for (int i = 0; i < N; i++) x[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Pinned boundary/center edge values. */
    x[0] = -128; x[1] = -1; x[2] = 0; x[3] = 1; x[4] = 127;
    /* A run of identical values (LUT-gather correctness under repeated
     * indices, spanning a vector-length boundary). */
    for (int i = 100; i < 140; i++) x[i] = 42;
    /* Far-from-zero pinned values where SiLU's mirror-asymmetry is large
     * (SiLU is NOT odd/even in x -- near 0 it's close to x/2, which looks
     * deceptively symmetric; these are chosen far from 0 to guarantee any
     * "negate before index" bug is caught). */
    x[150] = -100; x[151] = 80;
    /* Tail region (last 104 elements, i.e. indices 896..999). */
    for (int i = N - 104; i < N; i++) x[i] = (int8_t)((i * 7) - 50);

    for (int i = 0; i < N; i++) {
        uint8_t idx = (uint8_t)x[i];
        ref[i] = lut[idx];
    }

    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N, lut); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) fb = i;
        }
    }
    hvx_report(errors, N, fb,
               fb >= 0 ? (long)out[fb] : 0L,
               fb >= 0 ? (long)ref[fb] : 0L);
    return errors ? 1 : 0;
}
