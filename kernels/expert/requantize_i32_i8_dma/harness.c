/* requantize_i32_i8_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int32->int8 requantize at large N (int32 read, int8 write =>
 * ~5 bytes/elem of DDR traffic). Correctness contract identical to
 * requantize_i32_i8 (round-half-away-from-zero, +zp, int8 saturation). A single
 * runtime (mult,shift,zp) set is passed; inputs are bounded to int16 range so
 * a*mult fits int32 (word lanes). The HVX reference is cross-checked against the
 * exact int64 scalar reference on a sample. Harness owns main() and maps an
 * identity VTCM translation before the timed call. Word->byte packing uses
 * order-preserving vpacke. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 262144   /* a[]=1MB int32 + out 256KB; kernel working set > 1MB -> DDR-bound */
#endif
#define MULT  5
#define SHIFT 8
#define ZP    3

static int32_t a[N]  HVX_ALIGN;
static int8_t  out[N] HVX_ALIGN;
static int8_t  ref[N] HVX_ALIGN;

static int8_t ref_scalar(int32_t ai, int32_t mult, int shift, int8_t zp) {
    int64_t v = (int64_t)ai * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* word-lane requant -> word vector holding values in [-128,127]. */
static inline HVX_Vector requant_word(HVX_Vector w, int mult, HVX_Vector vhalf, int shift,
                                      HVX_Vector vzp, HVX_Vector vlo, HVX_Vector vhi,
                                      HVX_Vector vzero) {
    HVX_Vector v    = Q6_Vw_vmpyi_VwRh(w, (mult & 0xFFFF) | ((mult & 0xFFFF) << 16));
    HVX_Vector absv = Q6_Vw_vabs_Vw(v);
    HVX_Vector sh   = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(absv, vhalf), shift);
    HVX_Vector negsh= Q6_Vw_vsub_VwVw(vzero, sh);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VwVw(vzero, v);
    HVX_Vector r    = Q6_V_vmux_QVV(neg, negsh, sh);
    r = Q6_Vw_vadd_VwVw(r, vzp);
    r = Q6_Vw_vmax_VwVw(r, vlo);   /* int8 sat low  = -128 */
    r = Q6_Vw_vmin_VwVw(r, vhi);   /* int8 sat high =  127 */
    return r;
}
static inline HVX_Vector pack4(HVX_Vector r0, HVX_Vector r1, HVX_Vector r2, HVX_Vector r3) {
    HVX_Vector h0 = Q6_Vh_vpacke_VwVw(r1, r0);
    HVX_Vector h1 = Q6_Vh_vpacke_VwVw(r3, r2);
    return Q6_Vb_vpacke_VhVh(h1, h0);
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fill a[] with int32 values bounded to [-32768,32767] (word lanes). */
    {
        const int wpv = sizeof(HVX_Vector) / 4;   /* 32 */
        int32_t vi[32], vs[32];
        for (int j = 0; j < 32; j++) { vi[j] = j*2731 + 5; vs[j] = 32*2731; }
        HVX_Vector cur = *(HVX_Vector *)vi, step = *(HVX_Vector *)vs;
        HVX_Vector mff = Q6_V_vsplat_R(0x0000FFFFu), v32768 = Q6_V_vsplat_R(32768u);
        int i = 0;
        for (; i + wpv <= N; i += wpv) {
            *(HVX_Vector *)(a + i) = Q6_Vw_vsub_VwVw(Q6_V_vand_VV(cur, mff), v32768);
            cur = Q6_Vw_vadd_VwVw(cur, step);
        }
        for (; i < N; i++) a[i] = (int32_t)((uint32_t)(i*2731+5) & 0xFFFFu) - 32768;
    }
    a[0]=32767; a[1]=-32768; a[2]=0; a[3]=6400; a[4]=13000; a[5]=-13000;

    int hv = (SHIFT > 0) ? (1 << (SHIFT - 1)) : 0;
    HVX_Vector vhalf = Q6_V_vsplat_R((uint32_t)hv);
    HVX_Vector vzp   = Q6_V_vsplat_R((uint32_t)(int32_t)ZP);
    HVX_Vector vlo   = Q6_V_vsplat_R((uint32_t)(int32_t)(-128));
    HVX_Vector vhi   = Q6_V_vsplat_R((uint32_t)(int32_t)(127));
    HVX_Vector vzero = Q6_V_vzero();
    {
        const int wpv = sizeof(HVX_Vector) / 4;    /* 32 */
        int i = 0;
        for (; i + 4*wpv <= N; i += 4*wpv) {
            HVX_Vector r0 = requant_word(*(const HVX_Vector*)(a+i+0*wpv), MULT, vhalf, SHIFT, vzp, vlo, vhi, vzero);
            HVX_Vector r1 = requant_word(*(const HVX_Vector*)(a+i+1*wpv), MULT, vhalf, SHIFT, vzp, vlo, vhi, vzero);
            HVX_Vector r2 = requant_word(*(const HVX_Vector*)(a+i+2*wpv), MULT, vhalf, SHIFT, vzp, vlo, vhi, vzero);
            HVX_Vector r3 = requant_word(*(const HVX_Vector*)(a+i+3*wpv), MULT, vhalf, SHIFT, vzp, vlo, vhi, vzero);
            *(HVX_Vector *)(ref + i) = pack4(r0, r1, r2, r3);
        }
        for (; i < N; i++) ref[i] = ref_scalar(a[i], MULT, SHIFT, ZP);
    }
    {
        int bad = 0;
        for (int i = 0; i < N; i += 89) if (ref[i] != ref_scalar(a[i], MULT, SHIFT, ZP)) { bad = 1; break; }
        for (int i = 0; i < 8; i++) if (ref[i] != ref_scalar(a[i], MULT, SHIFT, ZP)) bad = 1;
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N, (int32_t)MULT, SHIFT, (int8_t)ZP); });
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
