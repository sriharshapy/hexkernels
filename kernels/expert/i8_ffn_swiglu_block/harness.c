/* i8_ffn_swiglu_block harness (v5, L3 -- SwiGLU gated FFN block).
 * SwiGLU = gate=X.Wg, up=X.Wu (two projections), H = SiLU(gate) * up (SiLU-gated
 * elementwise product), out = H.Wd (down projection): the gated feed-forward layer
 * of a modern transformer. The harness owns main(): builds a runtime SiLU LUT
 * (hardswish approximation, pure integer), fills seeded inputs, constructs a
 * zero-column-sum down-weight Wd (so the down-proj int8 activation zero-point
 * contributes no per-column offset and every requant field stays exact), computes
 * the reference via HVX vrmpy (a naive triple-loop scalar reference would blow the
 * sim budget), cross-checks it against an INDEPENDENT full scalar SwiGLU golden on
 * row 0, poisons the output, times the candidate (kernel-only pcycles), bit-exact
 * compares. Enables the HMX context AND an identity VTCM translation. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef S
#define S SW_S
#endif
#ifndef D
#define D SW_D
#endif
#ifndef DFF
#define DFF SW_DFF
#endif

static uint8_t  X  [S*D]     HVX_ALIGN;   /* activations uint8 0..3        */
static int8_t   Wg [D*DFF]   HVX_ALIGN;   /* gate weights int8             */
static int8_t   Wu [D*DFF]   HVX_ALIGN;   /* up   weights int8             */
static int8_t   Wgt[DFF*D]   HVX_ALIGN;   /* Wg transposed for vrmpy       */
static int8_t   Wut[DFF*D]   HVX_ALIGN;   /* Wu transposed for vrmpy       */
static int8_t   Wd [DFF*D]   HVX_ALIGN;   /* down weights int8 (ternary, cols zero-sum) */
static int8_t   Wdt[D*DFF]   HVX_ALIGN;   /* Wd transposed for vrmpy       */
static int32_t  bd [D]       HVX_ALIGN;
static uint8_t  Hu [S*DFF]   HVX_ALIGN;   /* reference gated activation (uint8) */
static int8_t   out[S*D]     HVX_ALIGN;
static int8_t   ref[S*D]     HVX_ALIGN;
static uint8_t  silu_lut[256] HVX_ALIGN;

static inline int32_t hreduce32(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* SiLU LUT (hardswish approximation, index = g+128, g in -128..127):
     * x = g/16; hardswish(x) = x*clamp(x+3,0,6)/6; store round(hardswish*16) as
     * int8. hardswish*16 = g*clamp(g+48,0,96)/96. Pure integer, deterministic,
     * passed to the kernel so SiLU is bit-exact by definition. */
    for (int idx = 0; idx < 256; idx++) {
        int g = idx - 128;
        int c = g + 48; if (c < 0) c = 0; if (c > 96) c = 96;
        int num = g * c;
        int hs = (num >= 0) ? (num + 48) / 96 : -((-num + 48) / 96);   /* round to nearest */
        silu_lut[idx] = (uint8_t)(signed char)sw_clamp_i8(hs);
    }

    uint32_t s = 0x3F17A9C5u;
    /* X in 0..3; Wg,Wu in -3..3.  gate/up: |acc| <= D*3*3 = 576 -> *17/16 = 612 < 2048. */
    for (int i = 0; i < S*D;   i++) X[i]  = (uint8_t)(hvx_lcg(&s) % 4);
    for (int i = 0; i < D*DFF; i++) Wg[i] = (int8_t)((int)((hvx_lcg(&s) >> 11) % 7) - 3);
    for (int i = 0; i < D*DFF; i++) Wu[i] = (int8_t)((int)((hvx_lcg(&s) >> 11) % 7) - 3);
    for (int j = 0; j < D;     j++) bd[j] = (int32_t)((int)((hvx_lcg(&s) >> 11) % 513) - 256);
    bd[0] = -4000; bd[1] = 4000;    /* int8-saturate-low and -high output columns */

    /* Down weight Wd: ternary {-1,0,1}, then force each COLUMN to sum to zero so
     * the down-proj activation zero-point (SW_ZP) contributes no offset -> the
     * accumulator equals the true gated.Wd and the 12-bit field stays exact. */
    for (int j = 0; j < DFF; j++)
        for (int m = 0; m < D; m++)
            Wd[j*D + m] = (int8_t)((int)((hvx_lcg(&s) >> 11) % 3) - 1);   /* -1..1 */
    for (int m = 0; m < D; m++) {
        int csum = 0;
        for (int j = 0; j < DFF; j++) csum += Wd[j*D + m];
        int j = 0;
        while (csum != 0) {
            int8_t *w = &Wd[j*D + m];
            if (csum > 0 && *w > -1) { (*w)--; csum--; }
            else if (csum < 0 && *w <  1) { (*w)++; csum++; }
            j++; if (j >= DFF) j = 0;
        }
    }

    /* transposes for contiguous vrmpy dots */
    for (int k = 0; k < D;   k++) for (int j = 0; j < DFF; j++) Wgt[j*D + k]   = Wg[k*DFF + j];
    for (int k = 0; k < D;   k++) for (int j = 0; j < DFF; j++) Wut[j*D + k]   = Wu[k*DFF + j];
    for (int k = 0; k < DFF; k++) for (int m = 0; m < D;   m++) Wdt[m*DFF + k] = Wd[k*D + m];

    const HVX_VectorPred dn  = Q6_Q_vsetq_R(D);     /* first D bytes valid (gate/up reduce) */
    const HVX_Vector zero = Q6_V_vzero();

    /* gate/up matmuls (HVX vrmpy) -> scale -> SiLU gate -> product -> Hu */
    for (int i = 0; i < S; i++) {
        HVX_Vector vx = Q6_V_vmux_QVV(dn, *(const HVX_UVector *)(X + i*D), zero);
        for (int j = 0; j < DFF; j++) {
            HVX_Vector vwg = Q6_V_vmux_QVV(dn, *(const HVX_UVector *)(Wgt + j*D), zero);
            HVX_Vector vwu = Q6_V_vmux_QVV(dn, *(const HVX_UVector *)(Wut + j*D), zero);
            int gacc = (int)hreduce32(Q6_Vw_vrmpyacc_VwVbVb(zero, vx, vwg));
            int uacc = (int)hreduce32(Q6_Vw_vrmpyacc_VwVbVb(zero, vx, vwu));
            int gq = sw_scale(hvx_hmx_requant_0x40(gacc), SW_SG);
            int uq = sw_scale(hvx_hmx_requant_0x40(uacc), SW_SU);
            int silu_g = (signed char)silu_lut[gq + 128];
            Hu[i*DFF + j] = sw_hu(silu_g, uq);
        }
    }
    /* down matmul (HVX vrmpy) -> requant + bias -> saturate.  DFF==128 => one vrmpy. */
    for (int i = 0; i < S; i++)
        for (int m = 0; m < D; m++) {
            HVX_Vector vh = *(const HVX_UVector *)(Hu + i*DFF);
            HVX_Vector vw = *(const HVX_UVector *)(Wdt + m*DFF);
            int acc = (int)hreduce32(Q6_Vw_vrmpyacc_VwVbVb(zero, vh, vw));
            ref[i*D + m] = sw_sat_i8((sw_sx12(hvx_hmx_requant_0x40(acc)) + bd[m]) >> SW_SO);
        }

    /* INDEPENDENT full scalar SwiGLU golden on row 0 (guards the vrmpy semantics). */
    {
        uint8_t Hus[DFF];
        for (int j = 0; j < DFF; j++) {
            int gacc = 0, uacc = 0;
            for (int k = 0; k < D; k++) {
                gacc += (int)X[0*D + k] * (int)Wg[k*DFF + j];
                uacc += (int)X[0*D + k] * (int)Wu[k*DFF + j];
            }
            int gq = sw_scale(hvx_hmx_requant_0x40(gacc), SW_SG);
            int uq = sw_scale(hvx_hmx_requant_0x40(uacc), SW_SU);
            int silu_g = (signed char)silu_lut[gq + 128];
            Hus[j] = sw_hu(silu_g, uq);
        }
        for (int m = 0; m < D; m++) {
            int acc = 0;
            for (int j = 0; j < DFF; j++) acc += (int)Hus[j] * (int)Wd[j*D + m];
            int gv = (int)sw_sat_i8((sw_sx12(hvx_hmx_requant_0x40(acc)) + bd[m]) >> SW_SO);
            if (gv != (int)ref[0*D + m]) {
                printf("HVXENV_REFCHECK_FAIL m=%d scalar=%d vrmpy=%d\n", m, gv, (int)ref[m]);
                return 2;
            }
        }
    }

    for (int i = 0; i < S*D; i++) *((volatile signed char *)&out[i]) = (signed char)0xA5; /* poison */

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(X, Wg, Wu, Wd, bd, silu_lut, out, S, D, DFF); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < S*D; i++) {
        int g = (int)out[i], e = (int)ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = g; expv = e; } }
    }
    hvx_report(errors, S*D, fb, gotv, expv);
    return errors ? 1 : 0;
}
