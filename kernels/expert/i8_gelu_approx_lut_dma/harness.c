/* i8_gelu_approx_lut_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8 GELU-approx (requant index + 256-entry LUT) at large N
 * (1B read + 1B write). Correctness contract identical to i8_gelu_approx_lut:
 * idx=clamp(in*scale/64+128,0,255) (C trunc toward zero); out=lut[idx]. A single
 * runtime (scale,lut) is passed; the kernel must still read both. The HVX
 * reference (halfword-lane index compute + 128B vlut32 8-pass table) is
 * cross-checked against the exact scalar reference on a sample. Harness owns
 * main() and maps an identity VTCM translation before the timed call. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 524288
#endif
#define SCALE 32

static int8_t in[N]  HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;
static int8_t lut[256] HVX_ALIGN;

static int clamp_idx(int x, int scale) {
    int idx = (x * scale) / 64 + 128;      /* C trunc toward zero */
    if (idx < 0) idx = 0; if (idx > 255) idx = 255;
    return idx;
}

static void lut_build(const int8_t *L, int8_t *tv0, int8_t *tv1) {
    for (int s = 0; s < 32; s++) {
        tv0[2*s]   = L[s];     tv0[64+2*s] = L[32+s];
        tv0[2*s+1] = L[64+s];  tv0[65+2*s] = L[96+s];
        tv1[2*s]   = L[128+s]; tv1[64+2*s] = L[160+s];
        tv1[2*s+1] = L[192+s]; tv1[65+2*s] = L[224+s];
    }
}
static inline HVX_Vector lut_apply(HVX_Vector vin, HVX_Vector T0, HVX_Vector T1) {
    HVX_Vector a = Q6_Vb_vlut32_VbVbR(vin, T0, 0);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T0, 1);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T0, 2);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T0, 3);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T1, 4);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T1, 5);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T1, 6);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T1, 7);
    return a;
}
/* halfword-lane requant index: clamp(x*scale/64 + 128, 0, 255). */
static inline HVX_Vector gelu_idx_half(HVX_Vector hv, HVX_Vector vscale,
                                       HVX_Vector v128, HVX_Vector v255, HVX_Vector vzero) {
    HVX_Vector v = Q6_Vh_vmpyi_VhVh(hv, vscale);         /* x*scale, fits int16 */
    HVX_Vector absv = Q6_Vh_vabs_Vh(v);
    HVX_Vector q = Q6_Vh_vasr_VhR(absv, 6);              /* |v|/64 */
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);
    HVX_Vector qs = Q6_V_vmux_QVV(neg, Q6_Vh_vsub_VhVh(vzero, q), q);  /* toward zero */
    HVX_Vector idx = Q6_Vh_vadd_VhVh(qs, v128);
    idx = Q6_Vh_vmax_VhVh(idx, vzero);
    idx = Q6_Vh_vmin_VhVh(idx, v255);
    return idx;
}
/* one byte-vector -> natural-order byte index vector. */
static inline HVX_Vector gelu_idx_bytevec(HVX_Vector xb, HVX_Vector vscale,
                                          HVX_Vector v128, HVX_Vector v255, HVX_Vector vzero) {
    HVX_VectorPair xh = Q6_Wh_vsxt_Vb(xb);
    HVX_Vector ilo = gelu_idx_half(Q6_V_lo_W(xh), vscale, v128, v255, vzero);
    HVX_Vector ihi = gelu_idx_half(Q6_V_hi_W(xh), vscale, v128, v255, vzero);
    HVX_VectorPair nat = Q6_W_vshuff_VVR(ihi, ilo, -2);
    return Q6_Vb_vpacke_VhVh(Q6_V_hi_W(nat), Q6_V_lo_W(nat));  /* low byte, natural */
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x5E1A2Fu;
    {
        const int vlen = sizeof(HVX_Vector);
        int8_t vi[128], vs[128];
        for (int j = 0; j < 128; j++) { vi[j] = (int8_t)(j*13+1); vs[j] = (int8_t)(128*13); }
        HVX_Vector cur = *(HVX_Vector *)vi, step = *(HVX_Vector *)vs;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(in + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) in[i] = (int8_t)(i*13+1);
    }
    in[0]=-128; in[1]=127; in[2]=0; in[3]=-1; in[4]=-127; in[5]=126;
    for (int k = 0; k < 256; k++) { s = s*1664525u + 1013904223u; lut[k] = (int8_t)(s >> 24); }

    HVX_Vector vscale = Q6_V_vsplat_R(((uint32_t)(SCALE & 0xFFFF) << 16) | (SCALE & 0xFFFF));
    HVX_Vector v128   = Q6_V_vsplat_R(0x00800080u);
    HVX_Vector v255   = Q6_V_vsplat_R(0x00FF00FFu);
    HVX_Vector vzero  = Q6_V_vzero();

    int8_t tv0[128] HVX_ALIGN, tv1[128] HVX_ALIGN;
    lut_build(lut, tv0, tv1);
    HVX_Vector T0 = *(HVX_Vector *)tv0, T1 = *(HVX_Vector *)tv1;
    {
        const int vlen = sizeof(HVX_Vector);
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            HVX_Vector idx = gelu_idx_bytevec(*(const HVX_Vector *)(in + i), vscale, v128, v255, vzero);
            *(HVX_Vector *)(ref + i) = lut_apply(idx, T0, T1);
        }
        for (; i < N; i++) ref[i] = lut[clamp_idx((int)in[i], SCALE)];
    }
    {
        int bad = 0;
        for (int i = 0; i < N; i += 89) if (ref[i] != lut[clamp_idx((int)in[i], SCALE)]) { bad = 1; break; }
        for (int i = 0; i < 8; i++) if (ref[i] != lut[clamp_idx((int)in[i], SCALE)]) bad = 1;
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
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N, lut, (int8_t)SCALE); });
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
