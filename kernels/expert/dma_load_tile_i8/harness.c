/* dma_load_tile_i8 harness (v6, Group C: dma + vtcm, load path).
 * Tile-staged identity copy at large N: out[i] = a[i]. Harness owns main();
 * seeds deterministic input (HVX ramp fill), poisons out to 0xA5, maps VTCM
 * identity so a DMA-load candidate can stage tiles on-chip, times the
 * candidate (kernel-only pcycles), bit-exact compares against a[].
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 786532   /* 48 full 16KB tiles + 100-byte remainder; ~768KB > L2 */
#endif

static int8_t a[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill of a[] with a deterministic ramp. */
    {
        const int vlen = sizeof(HVX_Vector);
        int8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (int8_t)(j * 7 + 3); v_step[j] = (int8_t)(128 * 7); }
        HVX_Vector cur = *(HVX_Vector *)v_init, step = *(HVX_Vector *)v_step;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(a + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) a[i] = (int8_t)(i * 7 + 3);
    }
    /* Boundary values injected. */
    a[0] = 127; a[1] = -128; a[N-2] = -128; a[N-1] = 127;

    /* Poison output. */
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

    /* HVX bulk verify against a[] (out must equal a[] exactly). */
    int errors = 0, fb = -1;
    {
        const int vlen = sizeof(HVX_Vector);
        int mismatch = 0, i = 0;
        for (; i + vlen <= N && !mismatch; i += vlen) {
            HVX_VectorPred eq = Q6_Q_vcmp_eq_VbVb(*(HVX_Vector *)(out + i), *(HVX_Vector *)(a + i));
            int8_t tmp[128] HVX_ALIGN;
            *(HVX_Vector *)tmp = Q6_V_vand_QR(eq, -1);
            for (int j = 0; j < vlen; j++) if ((uint8_t)tmp[j] != 0xFF) { mismatch = 1; break; }
        }
        for (; i < N; i++) if (out[i] != a[i]) { mismatch = 1; break; }
        if (mismatch)
            for (int i2 = 0; i2 < N; i2++) if (out[i2] != a[i2]) { errors++; if (fb < 0) fb = i2; }
    }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)a[fb]:0);
    return errors ? 1 : 0;
}
