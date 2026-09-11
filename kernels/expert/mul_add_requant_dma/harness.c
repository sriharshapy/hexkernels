/* mul_add_requant_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound fused multiply-add requantize (a*b + BIAS -> int8) at large N.
 * Inherits v4 mul_add_requant with the bias array baked to scalar BIAS and requant
 * params baked (BIAS=50, MULT=3, SHIFT=1, ZP=0). Harness owns main() and maps an
 * identity VTCM translation before the timed call. Inputs bounded to [-15,15] so
 * every intermediate fits int16 (halfword-lane compute). HVX reference cross-checked
 * against the int64 scalar golden on a sample.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 393216   /* 2 int32 inputs (1.5MB each) + int8 out -> DDR-bound */
#endif
#define BIAS  50
#define MULT  3
#define SHIFT 1
#define ZP    0

static int32_t a[N]   HVX_ALIGN;
static int32_t b[N]   HVX_ALIGN;
static int8_t  out[N] HVX_ALIGN;
static int8_t  ref[N] HVX_ALIGN;

static int8_t ref_scalar(int32_t ax, int32_t bx) {
    int64_t fma = (int64_t)ax * (int64_t)bx + (int64_t)BIAS;
    int64_t v   = fma * (int64_t)MULT;
    int64_t half = (SHIFT > 0) ? ((int64_t)1 << (SHIFT - 1)) : 0;
    int64_t r   = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
    r += ZP;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* halfword-domain requant of one int16 vector p (= a*b + BIAS). */
static inline HVX_Vector requant_half(HVX_Vector p, HVX_Vector vmult, HVX_Vector vhalf,
                                      int shift, HVX_Vector vzp, HVX_Vector vzero) {
    HVX_Vector v = Q6_Vh_vmpyi_VhVh(p, vmult);
    HVX_Vector absv = Q6_Vh_vabs_Vh(v);
    HVX_Vector sh = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absv, vhalf), shift);
    HVX_Vector negsh = Q6_Vh_vsub_VhVh(vzero, sh);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);
    HVX_Vector r = Q6_V_vmux_QVV(neg, negsh, sh);
    return Q6_Vh_vadd_VhVh(r, vzp);
}

/* 128 outputs: narrow a,b to int16, multiply, +BIAS, requant, pack to bytes. */
static inline HVX_Vector fma_block(const int32_t *ap, const int32_t *bp,
                                   HVX_Vector vmult, HVX_Vector vhalf, HVX_Vector vzp,
                                   HVX_Vector vbias, HVX_Vector vzero) {
    HVX_Vector a0=*(const HVX_Vector*)(ap), a1=*(const HVX_Vector*)(ap+32);
    HVX_Vector a2=*(const HVX_Vector*)(ap+64), a3=*(const HVX_Vector*)(ap+96);
    HVX_Vector b0=*(const HVX_Vector*)(bp), b1=*(const HVX_Vector*)(bp+32);
    HVX_Vector b2=*(const HVX_Vector*)(bp+64), b3=*(const HVX_Vector*)(bp+96);
    HVX_Vector ha0 = Q6_Vh_vpack_VwVw_sat(a1, a0), ha1 = Q6_Vh_vpack_VwVw_sat(a3, a2);
    HVX_Vector hb0 = Q6_Vh_vpack_VwVw_sat(b1, b0), hb1 = Q6_Vh_vpack_VwVw_sat(b3, b2);
    HVX_Vector p0 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(ha0, hb0), vbias);
    HVX_Vector p1 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(ha1, hb1), vbias);
    HVX_Vector r0 = requant_half(p0, vmult, vhalf, SHIFT, vzp, vzero);
    HVX_Vector r1 = requant_half(p1, vmult, vhalf, SHIFT, vzp, vzero);
    return Q6_Vb_vpack_VhVh_sat(r1, r0);
}

static void fill_small(int32_t *dst, int m, int off) {
    int32_t pat[32];
    for (int j = 0; j < 32; j++) pat[j] = (int32_t)(((j*m + off) % 31) - 15); /* [-15,15] */
    HVX_Vector vp = *(HVX_Vector *)pat;
    int i = 0;
    for (; i + 32 <= N; i += 32) *(HVX_Vector *)(dst + i) = vp;
    for (; i < N; i++) dst[i] = pat[i & 31];
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    fill_small(a, 7, 1);
    fill_small(b, 13, 5);
    a[0]=15;  b[0]=15;   /* 225+50=275*3=825>>1=412 -> sat +127 */
    a[1]=15;  b[1]=-15;  /* -225+50=-175*3=-525>>1=-262 -> sat -128 */
    a[2]=0;   b[2]=7;    /* fma=50, 50*3=150>>1=75 */
    a[3]=1;   b[3]=1;    /* fma=51, 153>>1=76 (306.../2 ... 153/2=76.5 -> tie -> 77? check) */
    a[4]=-1;  b[4]=1;    /* fma=49, 147>>1=73 (73.5 tie -> 74) */

    HVX_Vector vmult = Q6_Vh_vsplat_R(MULT);
    HVX_Vector vhalf = Q6_Vh_vsplat_R((SHIFT > 0) ? (1 << (SHIFT - 1)) : 0);
    HVX_Vector vzp   = Q6_Vh_vsplat_R(ZP);
    HVX_Vector vbias = Q6_Vh_vsplat_R(BIAS);
    HVX_Vector vzero = Q6_V_vzero();

    {
        int i = 0;
        for (; i + 128 <= N; i += 128)
            *(HVX_Vector *)(ref + i) = fma_block(a+i, b+i, vmult, vhalf, vzp, vbias, vzero);
        for (; i < N; i++) ref[i] = ref_scalar(a[i], b[i]);
    }
    {
        int bad = 0;
        for (int i = 0; i < N; i += 89) if (ref[i] != ref_scalar(a[i], b[i])) { bad = 1; break; }
        for (int i = 0; i < 8; i++) if (ref[i] != ref_scalar(a[i], b[i])) bad = 1;
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
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
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
