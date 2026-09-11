/* i8_add_bias_sat_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8 saturating scalar-bias add at large N. Correctness
 * contract identical to i8_add_bias_sat: out[i]=sat8(x[i]+bias). A single runtime
 * bias is passed (chosen within int8 range so a signed saturating byte add is
 * bit-exact; the kernel must still read it). Harness owns main() and maps an
 * identity VTCM translation before the timed call; uses HVX for init/ref/verify. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 524288   /* 4096 vectors; kernel working set x+out = 1.0MB >= L2 -> DDR-bound */
#endif
#define BIAS 100

static int8_t x[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

static inline int8_t sat8_add(int8_t v, int32_t bias) {
    int32_t r = (int32_t)v + bias; if (r > 127) r = 127; if (r < -128) r = -128; return (int8_t)r;
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        const int vlen = sizeof(HVX_Vector);
        int8_t vi[128], vs[128];
        for (int j = 0; j < 128; j++) { vi[j] = (int8_t)(j*7+3); vs[j] = (int8_t)(128*7); }
        HVX_Vector cur = *(HVX_Vector *)vi, step = *(HVX_Vector *)vs;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(x + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) x[i] = (int8_t)(i*7+3);
    }
    /* Saturation discriminators for bias=100. */
    x[0]=27;   x[1]=28;   x[2]=127;  x[3]=-128; x[4]=0;   x[5]=-100;

    /* bias is within int8 range -> signed saturating byte add is bit-exact. */
    HVX_Vector vb = Q6_V_vsplat_R(0x01010101u * (uint8_t)(int8_t)BIAS);
    {
        const int vlen = sizeof(HVX_Vector);
        int i = 0;
        for (; i + vlen <= N; i += vlen)
            *(HVX_Vector *)(ref + i) = Q6_Vb_vadd_VbVb_sat(*(const HVX_Vector *)(x + i), vb);
        for (; i < N; i++) ref[i] = sat8_add(x[i], BIAS);
    }
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N, (int32_t)BIAS); });
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
