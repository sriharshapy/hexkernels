/* rescale_i16_i8_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound rescale int16 -> int8 at large N. Contract identical to v4
 * rescale_i16_i8 with params baked (MULT=3, SHIFT=4, ZP=0). Harness owns main() and
 * maps an identity VTCM translation before the timed call. Inputs span the int16
 * range (a*MULT computed in int32); HVX reference cross-checked vs the int32 scalar
 * golden on a sample.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 524288    /* int16 in (1MB) + int8 out (512KB) -> DDR-bound */
#endif
#define MULT  3
#define SHIFT 4
#define ZP    0

static int16_t a[N]   HVX_ALIGN;
static int8_t  out[N] HVX_ALIGN;
static int8_t  ref[N] HVX_ALIGN;

static int8_t ref_scalar(int16_t x) {
    int32_t v = (int32_t)x * MULT;
    int32_t half = (SHIFT > 0) ? (1 << (SHIFT - 1)) : 0;
    int32_t r = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
    r += (int32_t)ZP;
    if (r > 127)  r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

static inline HVX_Vector requant_word(HVX_Vector x, int32_t mult_r,
                                      HVX_Vector vhalf, int shift, HVX_Vector vzp) {
    HVX_Vector vm = Q6_Vw_vmpyi_VwRh(x, mult_r);
    HVX_Vector sg = Q6_Vw_vasr_VwR(vm, 31);
    HVX_Vector ab = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(vm, sg), sg);
    HVX_Vector r  = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(ab, vhalf), shift);
    r = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(r, sg), sg);
    return Q6_Vw_vadd_VwVw(r, vzp);
}

/* pack 128 outputs (2 int16 vecs -> widen -> requant -> byte vec). */
static inline HVX_Vector rescale_block(HVX_Vector h0, HVX_Vector h1, int32_t mult_r,
                                       HVX_Vector vhalf, HVX_Vector vzp) {
    HVX_VectorPair w01 = Q6_Ww_vunpack_Vh(h0);
    HVX_VectorPair w23 = Q6_Ww_vunpack_Vh(h1);
    HVX_Vector r0 = requant_word(Q6_V_lo_W(w01), mult_r, vhalf, SHIFT, vzp);
    HVX_Vector r1 = requant_word(Q6_V_hi_W(w01), mult_r, vhalf, SHIFT, vzp);
    HVX_Vector r2 = requant_word(Q6_V_lo_W(w23), mult_r, vhalf, SHIFT, vzp);
    HVX_Vector r3 = requant_word(Q6_V_hi_W(w23), mult_r, vhalf, SHIFT, vzp);
    HVX_Vector ph01 = Q6_Vh_vpack_VwVw_sat(r1, r0);
    HVX_Vector ph23 = Q6_Vh_vpack_VwVw_sat(r3, r2);
    return Q6_Vb_vpack_VhVh_sat(ph23, ph01);
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* HVX fill of int16 a[] (halfword stepping, wraps within int16). */
    {
        int16_t vi[64], vs[64];
        for (int j = 0; j < 64; j++) { vi[j] = (int16_t)(j*517 + 3); vs[j] = (int16_t)(64*517); }
        HVX_Vector cur = *(HVX_Vector *)vi, step = *(HVX_Vector *)vs;
        int i = 0;
        for (; i + 64 <= N; i += 64) { *(HVX_Vector *)(a + i) = cur; cur = Q6_Vh_vadd_VhVh(cur, step); }
        for (; i < N; i++) a[i] = (int16_t)(i*517 + 3);
    }
    a[0]=1000;  /* 1000*3=3000>>4=187 -> sat +127 */
    a[1]=-1000; /* -> sat -128 */
    a[2]=8;     /* 8*3=24, +8>>4=2 (24/16=1.5 tie -> 2) */
    a[3]=-8;    /* -> -2 */
    a[4]=0;

    uint16_t m16 = (uint16_t)(int16_t)MULT;
    int32_t  mult_r = (int32_t)((uint32_t)m16 | ((uint32_t)m16 << 16));
    HVX_Vector vhalf = Q6_V_vsplat_R((SHIFT > 0) ? (1 << (SHIFT - 1)) : 0);
    HVX_Vector vzp   = Q6_V_vsplat_R((int32_t)ZP);

    {
        int i = 0;
        for (; i + 128 <= N; i += 128)
            *(HVX_Vector *)(ref + i) = rescale_block(*(HVX_Vector*)(a+i), *(HVX_Vector*)(a+i+64),
                                                     mult_r, vhalf, vzp);
        for (; i < N; i++) ref[i] = ref_scalar(a[i]);
    }
    {
        int bad = 0;
        for (int i = 0; i < N; i += 89) if (ref[i] != ref_scalar(a[i])) { bad = 1; break; }
        for (int i = 0; i < 8; i++) if (ref[i] != ref_scalar(a[i])) bad = 1;
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }
    {
        const int vlen = 128;
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    {
        const int vlen = 128;
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
