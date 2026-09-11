/* u8_threshold_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound uint8 inclusive binary threshold at large N (fixed thresh=128).
 * out=(in>=128)?255:0. Harness owns main(); sets up an identity VTCM translation
 * before the timed call so a candidate may DMA DDR<->VTCM. Uses HVX for
 * init/ref/verify so total simulated cycles stay within budget.
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
#define THRESH 128

static uint8_t a[N]   HVX_ALIGN;
static uint8_t out[N] HVX_ALIGN;
static uint8_t ref[N] HVX_ALIGN;

static inline uint8_t thr(uint8_t x) { return (uint8_t)(x >= THRESH ? 255 : 0); }

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill of a[] (sweeps all 256 byte values). */
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
        for (; i < N; i++) a[i] = (uint8_t)(i*13+1);
    }
    /* Boundary values: at/below/above threshold + extremes. */
    a[0]=127; a[1]=128; a[2]=129; a[3]=0; a[4]=255; a[5]=128;

    /* Reference via HVX: >=128 <=> unsigned > 127. */
    {
        const int vlen = sizeof(HVX_Vector);
        HVX_Vector vt = Q6_Vb_vsplat_R(THRESH - 1);   /* 127 */
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            HVX_VectorPred p = Q6_Q_vcmp_gt_VubVub(*(const HVX_Vector *)(a + i), vt);
            *(HVX_Vector *)(ref + i) = Q6_V_vand_QR(p, -1);
        }
        for (; i < N; i++) ref[i] = thr(a[i]);
    }
    /* Independent scalar cross-check of the HVX reference on a sample. */
    {
        int bad = 0;
        for (int i = 0; i < N; i += 97) if (ref[i] != thr(a[i])) { bad = 1; break; }
        for (int i = 0; i < 6; i++) if (ref[i] != thr(a[i])) bad = 1;
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }
    /* Poison output. */
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
