/* dma_wait_i8 harness (v6, Group C: dma + vtcm, explicit poll/wait).
 * uint8 saturating add-constant at large N. Harness owns main(); seeds
 * deterministic input including near-255 boundary values, poisons out,
 * maps VTCM identity, times the candidate (kernel-only pcycles), bit-exact
 * compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 786532
#endif

static uint8_t a[N]   HVX_ALIGN;
static uint8_t out[N] HVX_ALIGN;
static uint8_t ref[N] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (uint8_t)(j * 3); v_step[j] = (uint8_t)(128 * 3); }
        HVX_Vector cur = *(HVX_Vector *)v_init, step = *(HVX_Vector *)v_step;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(a + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) a[i] = (uint8_t)(i * 3);
    }
    /* Saturation boundary values. */
    a[0] = 255; a[1] = 250; a[2] = 206; a[N-2] = 205; a[N-1] = 0;

    for (int i = 0; i < N; i++) { int v = (int)a[i] + 50; ref[i] = (uint8_t)(v > 255 ? 255 : v); }

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t poison_arr[128]; memset(poison_arr, 0xA5, 128);
        HVX_Vector vpois = *(HVX_Vector *)poison_arr;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vpois;
        for (; i < N; i++) out[i] = (uint8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
