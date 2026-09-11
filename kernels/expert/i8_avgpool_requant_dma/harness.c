/* i8_avgpool_requant_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Fused 2x2 average pool (signed int8, truncate toward zero) + requantize, at large
 * image size so the working set (in ~1MB) exceeds L2 -> DDR-bandwidth-bound.
 * Correctness contract identical to i8_avgpool_requant; 3 (mult,shift,zp) sets are
 * swept so a candidate must read the runtime params. Harness owns main(); maps an
 * identity VTCM translation before the timed call. Init/reference/verify HVX-ified;
 * the HVX reference is cross-checked against the exact int64 scalar golden.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef W
#define W 1024
#endif
#ifndef H
#define H 768
#endif
#define OW (W/2)
#define OH (H/2)

static int8_t in[W*H]    HVX_ALIGN;
static int8_t out[OW*OH] HVX_ALIGN;
static int8_t ref[OW*OH] HVX_ALIGN;

static const int32_t MULTS[]  = {  5, -3 };
static const int     SHIFTS[] = {  0,  2 };
static const int8_t  ZPS[]    = { 10, -5 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

static int8_t ref_element(int pool, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)pool * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Requant one int16 half-vector of pooled values (sequential lanes). */
static inline HVX_Vector requant_h(HVX_Vector pool, HVX_Vector vmult, HVX_Vector vhalf,
                                   int shift, HVX_Vector vzp, HVX_Vector vzero) {
    HVX_Vector v    = Q6_Vh_vmpyi_VhVh(pool, vmult);
    HVX_Vector absv = Q6_Vh_vabs_Vh(v);
    HVX_Vector sh   = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absv, vhalf), shift);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);
    HVX_Vector r    = Q6_V_vmux_QVV(neg, Q6_Vh_vsub_VhVh(vzero, sh), sh);
    return Q6_Vh_vadd_VhVh(r, vzp);
}

/* Shared HVX fused 2x2 avg-pool (trunc toward zero) + requant over `nrows` rows. */
static void hvx_avgpool_requant(const int8_t *ip, int8_t *op, int w, int nrows,
                                int32_t mult, int shift, int8_t zp) {
    int ow = w / 2;
    HVX_Vector vzero = Q6_V_vzero();
    HVX_Vector vmult = Q6_Vh_vsplat_R(mult);
    int hv = shift > 0 ? (1 << (shift - 1)) : 0;
    HVX_Vector vhalf = Q6_Vh_vsplat_R(hv);
    HVX_Vector vzp   = Q6_Vh_vsplat_R((int)zp);
    for (int oy = 0; oy < nrows / 2; oy++) {
        const int8_t *r0 = ip + (2*oy)   * w;
        const int8_t *r1 = ip + (2*oy+1) * w;
        int8_t *o = op + oy * ow;
        int X = 0;
        for (; X + 256 <= w; X += 256) {
            HVX_VectorPair d0 = Q6_W_vdeal_VVR(*(const HVX_Vector *)(r0 + X + 128), *(const HVX_Vector *)(r0 + X), -1);
            HVX_VectorPair d1 = Q6_W_vdeal_VVR(*(const HVX_Vector *)(r1 + X + 128), *(const HVX_Vector *)(r1 + X), -1);
            /* sign-extend even/odd columns to i16 (sequential halves) and sum 4 pixels */
            HVX_VectorPair e0 = Q6_Wh_vsxt_Vb(Q6_V_lo_W(d0));  /* even cols row0 */
            HVX_VectorPair o0 = Q6_Wh_vsxt_Vb(Q6_V_hi_W(d0));  /* odd  cols row0 */
            HVX_VectorPair e1 = Q6_Wh_vsxt_Vb(Q6_V_lo_W(d1));
            HVX_VectorPair o1 = Q6_Wh_vsxt_Vb(Q6_V_hi_W(d1));
            HVX_Vector slo = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(Q6_V_lo_W(e0), Q6_V_lo_W(o0)),
                                             Q6_Vh_vadd_VhVh(Q6_V_lo_W(e1), Q6_V_lo_W(o1)));
            HVX_Vector shi = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(Q6_V_hi_W(e0), Q6_V_hi_W(o0)),
                                             Q6_Vh_vadd_VhVh(Q6_V_hi_W(e1), Q6_V_hi_W(o1)));
            /* pool = trunc(sum/4) toward zero */
            HVX_Vector ql = Q6_Vuh_vlsr_VuhR(Q6_Vh_vabs_Vh(slo), 2);
            HVX_Vector qh = Q6_Vuh_vlsr_VuhR(Q6_Vh_vabs_Vh(shi), 2);
            HVX_Vector poollo = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(vzero, slo), Q6_Vh_vsub_VhVh(vzero, ql), ql);
            HVX_Vector poolhi = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(vzero, shi), Q6_Vh_vsub_VhVh(vzero, qh), qh);
            HVX_Vector rlo = requant_h(poollo, vmult, vhalf, shift, vzp, vzero);
            HVX_Vector rhi = requant_h(poolhi, vmult, vhalf, shift, vzp, vzero);
            *(HVX_Vector *)(o + X/2) = Q6_Vb_vasr_VhVhR_sat(rhi, rlo, 0);  /* i16->i8 sat, seq halves */
        }
        for (int ox = X/2; ox < ow; ox++) {
            int a=r0[2*ox], b=r0[2*ox+1], c=r1[2*ox], d=r1[2*ox+1];
            o[ox] = ref_element((a+b+c+d)/4, mult, shift, zp);
        }
    }
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        const int vlen = sizeof(HVX_Vector);
        int8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (int8_t)(j*11-50); v_step[j] = (int8_t)(128*11); }
        HVX_Vector cur = *(HVX_Vector *)v_init, step = *(HVX_Vector *)v_step;
        int i = 0;
        for (; i + vlen <= W*H; i += vlen) { *(HVX_Vector *)(in + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < W*H; i++) in[i] = (int8_t)(i*11-50);
    }
    /* Edge windows (row 0). */
    in[0]=40; in[1]=40; in[W]=40; in[W+1]=40;         /* avg 40 */
    in[2]=-50; in[3]=-50; in[W+2]=-50; in[W+3]=-50;   /* avg -50 */
    in[4]=3; in[5]=0; in[W+4]=0; in[W+5]=0;           /* 3/4 -> 0 (trunc toward zero) */
    in[6]=-3; in[7]=0; in[W+6]=0; in[W+7]=0;          /* -3/4 -> 0 (trunc toward zero) */
    in[8]=127; in[9]=127; in[W+8]=127; in[W+9]=127;   /* avg 127, requant saturate */

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int k = 0; k < NSETS; k++) {
        int32_t mult = MULTS[k]; int shift = SHIFTS[k]; int8_t zp = ZPS[k];
        hvx_avgpool_requant(in, ref, W, H, mult, shift, zp);
        /* cross-check reference vs scalar golden on a sample (incl edges). */
        {
            int bad = 0;
            for (int p = 0; p < OW*OH && !bad; p += 101) {
                int oy = p / OW, ox = p % OW;
                int a=in[(2*oy)*W+2*ox], b=in[(2*oy)*W+2*ox+1], c=in[(2*oy+1)*W+2*ox], d=in[(2*oy+1)*W+2*ox+1];
                if (ref[p] != ref_element((a+b+c+d)/4, mult, shift, zp)) bad = 1;
            }
            for (int ox = 0; ox < 10 && !bad; ox++) {
                int a=in[2*ox], b=in[2*ox+1], c=in[W+2*ox], d=in[W+2*ox+1];
                if (ref[ox] != ref_element((a+b+c+d)/4, mult, shift, zp)) bad = 1;
            }
            if (bad) { printf("HVXENV_REF_MISMATCH (harness bug) set=%d\n", k); return 2; }
        }
        for (int i = 0; i < OW*OH; i++) out[i] = (int8_t)0xA5;   /* poison */
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, W, H, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
        for (int i = 0; i < OW*OH; i++) {
            if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = k*OW*OH + i; gotv=(long)out[i]; expv=(long)ref[i]; } }
        }
    }
    hvx_report(errors, OW*OH*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
