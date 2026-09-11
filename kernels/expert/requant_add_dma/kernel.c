/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams both int32 inputs a[],b[] from DDR into VTCM via uDMA, computes the
 * residual-add requantize on the on-chip copies with HVX, DMAs the int8 results
 * back. Double-buffered: the next tile's inputs are prefetched (async DMA) while
 * the current tile computes, hiding DDR latency. Requant params baked
 * (MULT=5, SHIFT=3, ZP=0); round half away from zero; sat8 on the pack. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define MULT  5
#define SHIFT 3
#define ZP    0
#define T     4096            /* elements per tile */
#define TB_IN (T * 4)         /* int32 tile bytes */
#define TB_OUT (T)            /* int8  tile bytes */

static desc_t d_ina, d_inb, d_pfa, d_pfb, d_out;

static inline HVX_Vector requant_word(HVX_Vector x, int32_t mult_r,
                                      HVX_Vector vhalf, int shift, HVX_Vector vzp) {
    HVX_Vector vm = Q6_Vw_vmpyi_VwRh(x, mult_r);
    HVX_Vector sg = Q6_Vw_vasr_VwR(vm, 31);
    HVX_Vector ab = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(vm, sg), sg);
    HVX_Vector r  = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(ab, vhalf), shift);
    r = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(r, sg), sg);
    return Q6_Vw_vadd_VwVw(r, vzp);
}

static int8_t ref_scalar(int32_t ax, int32_t bx) {
    int64_t sum = (int64_t)ax + (int64_t)bx;
    int64_t v   = sum * (int64_t)MULT;
    int64_t half = (SHIFT > 0) ? ((int64_t)1 << (SHIFT - 1)) : 0;
    int64_t r   = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
    r += ZP;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

static inline void compute_tile(uint32_t va, uint32_t vb, uint32_t vo,
                                int32_t mult_r, HVX_Vector vhalf, HVX_Vector vzp) {
    HVX_Vector *pa = (HVX_Vector *)(uintptr_t)va;
    HVX_Vector *pb = (HVX_Vector *)(uintptr_t)vb;
    HVX_Vector *po = (HVX_Vector *)(uintptr_t)vo;
    for (int g = 0; g < T / 128; g++) {
        int base = g * 4;
        HVX_Vector s0 = Q6_Vw_vadd_VwVw(pa[base+0], pb[base+0]);
        HVX_Vector s1 = Q6_Vw_vadd_VwVw(pa[base+1], pb[base+1]);
        HVX_Vector s2 = Q6_Vw_vadd_VwVw(pa[base+2], pb[base+2]);
        HVX_Vector s3 = Q6_Vw_vadd_VwVw(pa[base+3], pb[base+3]);
        HVX_Vector r0 = requant_word(s0, mult_r, vhalf, SHIFT, vzp);
        HVX_Vector r1 = requant_word(s1, mult_r, vhalf, SHIFT, vzp);
        HVX_Vector r2 = requant_word(s2, mult_r, vhalf, SHIFT, vzp);
        HVX_Vector r3 = requant_word(s3, mult_r, vhalf, SHIFT, vzp);
        HVX_Vector ph01 = Q6_Vh_vpack_VwVw_sat(r1, r0);
        HVX_Vector ph23 = Q6_Vh_vpack_VwVw_sat(r3, r2);
        po[g] = Q6_Vb_vpack_VhVh_sat(ph23, ph01);
    }
}

void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n) {
    uint16_t m16 = (uint16_t)(int16_t)MULT;
    int32_t  mult_r = (int32_t)((uint32_t)m16 | ((uint32_t)m16 << 16));
    HVX_Vector vhalf = Q6_V_vsplat_R((SHIFT > 0) ? (1 << (SHIFT - 1)) : 0);
    HVX_Vector vzp   = Q6_V_vsplat_R((int32_t)ZP);

    const uint32_t va0 = VTCM_BASE,          va1 = VTCM_BASE + TB_IN;
    const uint32_t vb0 = VTCM_BASE + 2*TB_IN, vb1 = VTCM_BASE + 3*TB_IN;
    const uint32_t vo0 = VTCM_BASE + 4*TB_IN, vo1 = VTCM_BASE + 4*TB_IN + TB_OUT;

    int nfull = n / T;
    int rem   = n - nfull * T;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i], b[i]);
        return;
    }

    /* chain the two input DMAs into one linked descriptor list per dmstart. */
    d_inb.next = 0;                              d_inb.ctrl = TB_IN; d_inb.src = (uint32_t)(uintptr_t)b; d_inb.dst = vb0;
    d_ina.next = (uint32_t)(uintptr_t)&d_inb;    d_ina.ctrl = TB_IN; d_ina.src = (uint32_t)(uintptr_t)a; d_ina.dst = va0;
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

        compute_tile(cur_a, cur_b, cur_o, mult_r, vhalf, vzp);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = TB_OUT; d_out.src = cur_o;
        d_out.dst = (uint32_t)(uintptr_t)(out + c * T);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * T; i < nfull * T + rem; i++)
        out[i] = ref_scalar(a[i], b[i]);
}
