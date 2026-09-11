/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams both int32 inputs a[],b[] from DDR into VTCM via uDMA, computes the fused
 * multiply-add requantize (a*b + BIAS -> int8) on the on-chip copies with HVX, DMAs
 * the int8 results back. Double-buffered: next tile's inputs prefetched (async DMA)
 * while the current tile computes. Params baked (BIAS=50, MULT=3, SHIFT=1, ZP=0);
 * inputs bounded to [-15,15] -> halfword-lane compute; round half away; sat8 pack. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define BIAS  50
#define MULT  3
#define SHIFT 1
#define ZP    0
#define T     4096
#define TB_IN (T * 4)
#define TB_OUT (T)

static desc_t d_ina, d_inb, d_pfa, d_pfb, d_out;

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

static inline void compute_tile(uint32_t va, uint32_t vb, uint32_t vo,
                                HVX_Vector vmult, HVX_Vector vhalf, HVX_Vector vzp,
                                HVX_Vector vbias, HVX_Vector vzero) {
    HVX_Vector *pa = (HVX_Vector *)(uintptr_t)va;   /* 32 int32 per vector */
    HVX_Vector *pb = (HVX_Vector *)(uintptr_t)vb;
    HVX_Vector *po = (HVX_Vector *)(uintptr_t)vo;
    for (int g = 0; g < T / 128; g++) {
        int base = g * 4;
        HVX_Vector ha0 = Q6_Vh_vpack_VwVw_sat(pa[base+1], pa[base+0]);
        HVX_Vector ha1 = Q6_Vh_vpack_VwVw_sat(pa[base+3], pa[base+2]);
        HVX_Vector hb0 = Q6_Vh_vpack_VwVw_sat(pb[base+1], pb[base+0]);
        HVX_Vector hb1 = Q6_Vh_vpack_VwVw_sat(pb[base+3], pb[base+2]);
        HVX_Vector p0 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(ha0, hb0), vbias);
        HVX_Vector p1 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(ha1, hb1), vbias);
        HVX_Vector r0 = requant_half(p0, vmult, vhalf, SHIFT, vzp, vzero);
        HVX_Vector r1 = requant_half(p1, vmult, vhalf, SHIFT, vzp, vzero);
        po[g] = Q6_Vb_vpack_VhVh_sat(r1, r0);
    }
}

void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n) {
    HVX_Vector vmult = Q6_Vh_vsplat_R(MULT);
    HVX_Vector vhalf = Q6_Vh_vsplat_R((SHIFT > 0) ? (1 << (SHIFT - 1)) : 0);
    HVX_Vector vzp   = Q6_Vh_vsplat_R(ZP);
    HVX_Vector vbias = Q6_Vh_vsplat_R(BIAS);
    HVX_Vector vzero = Q6_V_vzero();

    const uint32_t va0 = VTCM_BASE,          va1 = VTCM_BASE + TB_IN;
    const uint32_t vb0 = VTCM_BASE + 2*TB_IN, vb1 = VTCM_BASE + 3*TB_IN;
    const uint32_t vo0 = VTCM_BASE + 4*TB_IN, vo1 = VTCM_BASE + 4*TB_IN + TB_OUT;

    int nfull = n / T;
    int rem   = n - nfull * T;

    if (nfull == 0) { for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i], b[i]); return; }

    d_inb.next = 0;                           d_inb.ctrl = TB_IN; d_inb.src = (uint32_t)(uintptr_t)b; d_inb.dst = vb0;
    d_ina.next = (uint32_t)(uintptr_t)&d_inb; d_ina.ctrl = TB_IN; d_ina.src = (uint32_t)(uintptr_t)a; d_ina.dst = va0;
    Q6_dmstart_A(&d_ina); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_a = (c & 1) ? va1 : va0;
        uint32_t cur_b = (c & 1) ? vb1 : vb0;
        uint32_t cur_o = (c & 1) ? vo1 : vo0;
        uint32_t nxt_a = (c & 1) ? va0 : va1;
        uint32_t nxt_b = (c & 1) ? vb0 : vb1;

        if (c + 1 < nfull) {
            const int32_t *na = a + (c + 1) * T;
            const int32_t *nb = b + (c + 1) * T;
            d_pfb.next = 0;                           d_pfb.ctrl = TB_IN; d_pfb.src = (uint32_t)(uintptr_t)nb; d_pfb.dst = nxt_b;
            d_pfa.next = (uint32_t)(uintptr_t)&d_pfb; d_pfa.ctrl = TB_IN; d_pfa.src = (uint32_t)(uintptr_t)na; d_pfa.dst = nxt_a;
            Q6_dmstart_A(&d_pfa);
        }

        compute_tile(cur_a, cur_b, cur_o, vmult, vhalf, vzp, vbias, vzero);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = TB_OUT; d_out.src = cur_o;
        d_out.dst = (uint32_t)(uintptr_t)(out + c * T);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * T; i < nfull * T + rem; i++) out[i] = ref_scalar(a[i], b[i]);
}
