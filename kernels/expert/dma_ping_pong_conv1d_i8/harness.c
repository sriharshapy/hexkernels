/* dma_ping_pong_conv1d_i8 harness (v6, Group C: halo'd dma double-buffer).
 * 3-tap FIR conv1d, edge-replicated boundaries, bias+shift requant, at large
 * N. Harness owns main(): seeds deterministic (effectively random, hence
 * asymmetric) int8 inputs via hvx_lcg so a left/right tap swap is a robust
 * discriminator; computes the exact sign-aware scalar reference with
 * clamp-to-edge; poisons out; maps VTCM identity; times the candidate
 * (kernel-only pcycles); bit-exact compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 1200000  /* 1.2MB int8 > L2 (int8 dtype needs 4x the element count of
                    * the int32 tasks to hit the same byte footprint); 73 full
                    * 16KB tiles + 3968 remainder */
#endif

static int8_t a[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

static const int8_t W[3] = {2, 5, 3};
#define BIAS  10
#define SHIFT 4

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0xC0FFEE42u;
    for (int i = 0; i < N; i++) a[i] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Asymmetric boundary edge cases (a[i-1] != a[i+1] guaranteed here). */
    a[0] = 127; a[1] = -128; a[2] = 64; a[3] = -64;
    a[N-2] = -100; a[N-1] = 100;
    a[16383] = 30; a[16384] = -30;   /* around the first tile boundary */

    int32_t half = (SHIFT > 0) ? (1 << (SHIFT - 1)) : 0;
    for (int i = 0; i < N; i++) {
        int32_t left   = a[clampi(i - 1, 0, N - 1)];
        int32_t center = a[i];
        int32_t right  = a[clampi(i + 1, 0, N - 1)];
        int32_t t = (int32_t)W[0]*left + (int32_t)W[1]*center + (int32_t)W[2]*right + BIAS;
        int64_t abst = t < 0 ? -(int64_t)t : (int64_t)t;
        int64_t sh = (abst + half) >> SHIFT;
        int64_t r = t < 0 ? -sh : sh;
        if (r > 127) r = 127; if (r < -128) r = -128;
        ref[i] = (int8_t)r;
    }

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t poison_arr[128]; for (int j = 0; j < 128; j++) poison_arr[j] = 0xA5;
        HVX_Vector vpois = *(HVX_Vector *)poison_arr;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vpois;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, W, BIAS, SHIFT, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
