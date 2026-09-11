/* dma_load_broadcast_i8 harness (v6, Group C: dma double-buffer + broadcast
 * operand). Saturating int8 add of a large stream with a small (128B)
 * cyclically-broadcast operand. Harness owns main(): seeds deterministic
 * inputs with saturation-boundary values, maps VTCM identity, times the
 * candidate (kernel-only pcycles), bit-exact compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 700000   /* a+out ~1.37MB > L2 */
#endif

static int8_t a[N]   HVX_ALIGN;
static int8_t op[128] HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        uint32_t s = 0x77335511u;
        for (int i = 0; i < N; i++) a[i] = (int8_t)hvx_lcg(&s);
        for (int j = 0; j < 128; j++) op[j] = (int8_t)((j * 5 + 3) - 128);
    }
    /* Saturation boundary values. */
    a[0] = 127;  op[0] = 1;    /* -> saturate +127 */
    a[1] = -128; op[1] = -1;   /* -> saturate -128 */
    a[N-1] = 100; op[(N-1)%128] = 100;

    for (int i = 0; i < N; i++) {
        int32_t t = (int32_t)a[i] + (int32_t)op[i % 128];
        if (t > 127) t = 127; if (t < -128) t = -128;
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
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, op, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
