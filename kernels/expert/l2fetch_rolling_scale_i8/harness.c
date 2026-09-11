/* l2fetch_rolling_scale_i8 harness (v6, Group C: rolling l2fetch + unroll).
 * Fixed-point scale-by-1.5-with-round at large N (>1MB single stream).
 * Harness owns main(): seeds deterministic inputs, poisons out, times the
 * candidate (kernel-only pcycles), bit-exact compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 700000   /* a+out ~1.37MB > L2; 5468 full vectors + 96-byte tail */
#endif

static int8_t a[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

int main(void) {
    {
        uint32_t s = 0x2468ACE1u;
        for (int i = 0; i < N; i++) a[i] = (int8_t)hvx_lcg(&s);
    }
    /* Saturation boundary values. */
    a[0] = 127; a[1] = -128; a[2] = 100; a[3] = -100;
    a[N-2] = 127; a[N-1] = -128;

    for (int i = 0; i < N; i++) {
        int32_t t = (int32_t)a[i] * 3;
        t = (t + 1) >> 1;
        if (t > 127) t = 127;
        if (t < -128) t = -128;
        ref[i] = (int8_t)t;
    }

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t poison_arr[128]; memset(poison_arr, 0xA5, 128);
        HVX_Vector vpois = *(HVX_Vector *)poison_arr;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vpois;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
