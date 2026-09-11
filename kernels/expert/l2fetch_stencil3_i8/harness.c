/* l2fetch_stencil3_i8 harness (v6, Group C: rolling l2fetch + 1D stencil).
 * 3-tap weighted (1,2,1) edge-replicated stencil at large N (>1MB single
 * stream). Harness owns main(): seeds deterministic inputs, poisons out,
 * times the candidate (kernel-only pcycles), bit-exact compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 700000   /* a+out ~1.37MB > L2 */
#endif

static int8_t a[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

int main(void) {
    {
        uint32_t s = 0x13572468u;
        for (int i = 0; i < N; i++) a[i] = (int8_t)hvx_lcg(&s);
    }
    /* Saturation-forcing values near both ends. */
    a[0] = 127; a[1] = 127; a[2] = 127;
    a[N-1] = -128; a[N-2] = -128; a[N-3] = -128;

    for (int i = 0; i < N; i++) {
        int ip = (i > 0) ? i - 1 : 0;
        int in = (i < N - 1) ? i + 1 : N - 1;
        int32_t t = (int32_t)a[ip] + 2 * (int32_t)a[i] + (int32_t)a[in];
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
