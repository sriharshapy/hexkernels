/* i8_abs_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound saturating int8 abs at large N. Correctness contract identical
 * to i8_abs: |x| with -128 saturating to 127. Harness owns main(); sets up an
 * identity VTCM translation before the timed call so a candidate may DMA
 * DDR<->VTCM. Uses HVX for init/ref/verify so total simulated cycles stay within
 * the sim wall-clock budget.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 524288   /* 4096 vectors; in+out = 2*512KB = 1MB streamed -> DDR-bound */
#endif

static int8_t a[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill of a[]. */
    {
        const int vlen = sizeof(HVX_Vector);
        int8_t va_init[128], va_step[128];
        for (int j = 0; j < 128; j++) {
            va_init[j] = (int8_t)(j * 13 + 1);
            va_step[j] = (int8_t)(128 * 13);
        }
        HVX_Vector cur_a = *(HVX_Vector *)va_init, step_a = *(HVX_Vector *)va_step;
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            *(HVX_Vector *)(a + i) = cur_a;
            cur_a = Q6_Vb_vadd_VbVb(cur_a, step_a);
        }
        for (; i < N; i++) a[i] = (int8_t)(i*13+1);
    }
    /* Boundary values: catch missing -128 saturation. */
    a[0]=-128; a[1]=127; a[2]=0; a[3]=-1; a[4]=1; a[5]=-127;

    /* Reference via HVX saturating abs (same op as a correct candidate). */
    {
        const int vlen = sizeof(HVX_Vector);
        int i = 0;
        for (; i + vlen <= N; i += vlen)
            *(HVX_Vector *)(ref + i) = Q6_Vb_vabs_Vb_sat(*(const HVX_Vector *)(a + i));
        for (; i < N; i++) { int v = a[i]; if (v < 0) v = -v; if (v > 127) v = 127; ref[i] = (int8_t)v; }
    }
    /* Independent scalar cross-check of the HVX reference on a sample. */
    {
        int bad = 0;
        for (int i = 0; i < N; i += 97) { int v = a[i]; if (v < 0) v = -v; if (v > 127) v = 127; if (ref[i] != (int8_t)v) { bad = 1; break; } }
        for (int i = 0; i < 6; i++) { int v = a[i]; if (v < 0) v = -v; if (v > 127) v = 127; if (ref[i] != (int8_t)v) bad = 1; }
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }
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

    int errors = 0, fb = -1;
    {
        const int vlen = sizeof(HVX_Vector);
        int mismatch = 0, i = 0;
        for (; i + vlen <= N && !mismatch; i += vlen) {
            HVX_VectorPred eq = Q6_Q_vcmp_eq_VbVb(
                *(HVX_Vector *)(out + i), *(HVX_Vector *)(ref + i));
            int8_t tmp[128] HVX_ALIGN;
            *(HVX_Vector *)tmp = Q6_V_vand_QR(eq, -1);
            for (int j = 0; j < vlen; j++) if ((uint8_t)tmp[j] != 0xFF) { mismatch = 1; break; }
        }
        for (; i < N; i++) if (out[i] != ref[i]) { mismatch = 1; break; }
        if (mismatch)
            for (int i2 = 0; i2 < N; i2++) if (out[i2] != ref[i2]) { errors++; if (fb < 0) fb = i2; }
    }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
