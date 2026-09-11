/* i8_min_scalar_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8 elementwise min-with-scalar at large N. Correctness
 * contract identical to i8_min_scalar: out[i]=min(in[i],c), signed int8. A single
 * runtime c is passed (the kernel must still read it). Harness owns main() and
 * maps an identity VTCM translation before the timed call; uses HVX for
 * init/ref/verify so simulated cycles stay within the sim wall budget. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 524288   /* 4096 vectors; kernel working set in+out = 1.0MB >= L2 -> DDR-bound */
#endif
#define C 0        /* ceiling-at-zero clamp */

static int8_t in[N]  HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        const int vlen = sizeof(HVX_Vector);
        int8_t vi[128], vs[128];
        for (int j = 0; j < 128; j++) { vi[j] = (int8_t)(j*7+3); vs[j] = (int8_t)(128*7); }
        HVX_Vector cur = *(HVX_Vector *)vi, step = *(HVX_Vector *)vs;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(in + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) in[i] = (int8_t)(i*7+3);
    }
    in[0]=-128; in[1]=127; in[2]=0; in[3]=-1; in[4]=1;

    HVX_Vector vc = Q6_V_vsplat_R(0x01010101u * (uint8_t)(int8_t)C);
    {
        const int vlen = sizeof(HVX_Vector);
        int i = 0;
        for (; i + vlen <= N; i += vlen)
            *(HVX_Vector *)(ref + i) = Q6_Vb_vmin_VbVb(*(const HVX_Vector *)(in + i), vc);
        for (; i < N; i++) { int8_t x = in[i]; ref[i] = (x < (int8_t)C) ? x : (int8_t)C; }
    }
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N, (int8_t)C); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    {
        const int vlen = sizeof(HVX_Vector);
        int mismatch = 0, i = 0;
        for (; i + vlen <= N && !mismatch; i += vlen) {
            HVX_VectorPred eq = Q6_Q_vcmp_eq_VbVb(*(HVX_Vector *)(out + i), *(HVX_Vector *)(ref + i));
            int8_t tmp[128] HVX_ALIGN; *(HVX_Vector *)tmp = Q6_V_vand_QR(eq, -1);
            for (int j = 0; j < vlen; j++) if ((uint8_t)tmp[j] != 0xFF) { mismatch = 1; break; }
        }
        for (; i < N; i++) if (out[i] != ref[i]) { mismatch = 1; break; }
        if (mismatch) for (int i2 = 0; i2 < N; i2++) if (out[i2] != ref[i2]) { errors++; if (fb < 0) fb = i2; }
    }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
