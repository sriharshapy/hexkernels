/* dma_stream_requant_i32_i8 harness (v6, Group C: dma + vtcm double-buffer).
 * Requantize int32->int8 (sign-aware round-half-away-from-zero + zp) at
 * large N. Harness owns main(): seeds deterministic bounded inputs (kept
 * small enough that |a[i]|*mult fits int32, matching the HVX vmpyie path),
 * with explicit near-overflow-boundary and sign edge cases; computes the
 * exact sign-aware scalar reference; poisons out; maps VTCM identity; times
 * the candidate (kernel-only pcycles); bit-exact compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 300000   /* 1.2MB int32 in > L2; 73 full 16KB(=4096-elem) tiles + 992 remainder */
#endif

#define MULT  200
#define SHIFT 8
#define ZP    5

static int32_t a[N]   HVX_ALIGN;
static int8_t  out[N] HVX_ALIGN;
static int8_t  ref[N] HVX_ALIGN;

/* General fill is small-magnitude ([-150,150]) so the requantized result is
 * NOT already saturated for most elements -- that way "forgot zp" is a
 * total-coverage discriminator (every element off by exactly zp), not masked
 * by int8 saturation. A handful of explicit near-overflow-boundary elements
 * (below) separately exercise the |a[i]|*mult<2^31 safety margin. */
static inline int32_t clampv(uint32_t r) { return (int32_t)(r % 301u) - 150; }

static inline int8_t ref_requant(int32_t v, int32_t mult, int shift, int8_t zp) {
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t absv = (v < 0) ? -(int64_t)v : (int64_t)v;
    int64_t am   = absv * (int64_t)mult;
    int64_t sh   = (am + half) >> shift;
    int64_t r    = (v < 0) ? -sh : sh;
    r += zp;
    if (r > 127) r = 127; if (r < -128) r = -128;
    return (int8_t)r;
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0xC0FFEE42u;
    for (int i = 0; i < N; i++) a[i] = clampv(hvx_lcg(&s));
    /* Overflow-boundary + sign edge cases. */
    a[0] = 2000000; a[1] = -2000000; a[2] = 0; a[3] = 1; a[4] = -1;
    a[N-2] = 2000000; a[N-1] = -2000000;
    if (N > 299008) { a[299007] = 2000000; a[299008] = -2000000; }

    for (int i = 0; i < N; i++) ref[i] = ref_requant(a[i], MULT, SHIFT, ZP);

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t poison_arr[128]; for (int j = 0; j < 128; j++) poison_arr[j] = 0xA5;
        HVX_Vector vpois = *(HVX_Vector *)poison_arr;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vpois;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N, MULT, SHIFT, ZP); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
