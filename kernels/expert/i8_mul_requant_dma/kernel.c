/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams a[] and b[] from DDR into VTCM via uDMA, computes int8 multiply+requant
 * on the on-chip copies with HVX, and DMAs results back. Double-buffered: the next
 * input tiles are prefetched (async DMA) while the current tile computes, hiding
 * DDR latency on this bandwidth-bound task. Halfword-lane requant with
 * round-half-away sign-split (valid while prod*mult fits int16). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH 16384
static desc_t d_in_a, d_in_b, d_out, d_pf_a, d_pf_b;

static inline HVX_Vector mulrq_vec(HVX_Vector va, HVX_Vector vb,
                                   HVX_Vector vmult, HVX_Vector vhalf,
                                   HVX_Vector vzp, int shift, HVX_Vector vzero) {
    HVX_VectorPair ah = Q6_Wh_vsxt_Vb(va);
    HVX_VectorPair bh = Q6_Wh_vsxt_Vb(vb);
    HVX_Vector al[2] = { Q6_V_lo_W(ah), Q6_V_hi_W(ah) };
    HVX_Vector bl[2] = { Q6_V_lo_W(bh), Q6_V_hi_W(bh) };
    HVX_Vector res[2];
    for (int k = 0; k < 2; k++) {
        HVX_Vector prod = Q6_Vh_vmpyi_VhVh(al[k], bl[k]);
        HVX_Vector v    = Q6_Vh_vmpyi_VhVh(prod, vmult);
        HVX_Vector absv = Q6_Vh_vabs_Vh(v);
        HVX_Vector sh   = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absv, vhalf), shift);
        HVX_Vector negsh= Q6_Vh_vsub_VhVh(vzero, sh);
        HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);
        HVX_Vector r    = Q6_V_vmux_QVV(neg, negsh, sh);
        res[k] = Q6_Vh_vadd_VhVh(r, vzp);
    }
    return Q6_Vb_vasr_VhVhR_sat(res[1], res[0], 0);
}

static int8_t ref_scalar(int8_t ai, int8_t bi, int32_t mult, int shift, int8_t zp) {
    int32_t prod = (int32_t)ai * (int32_t)bi;
    int64_t v = (int64_t)prod * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp; if (r > 127) r = 127; if (r < -128) r = -128; return (int8_t)r;
}

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    uint32_t mw = ((uint32_t)(mult & 0xFFFF) << 16) | (mult & 0xFFFF);
    int hv = (shift > 0) ? (1 << (shift - 1)) : 0;
    uint32_t hw = ((uint32_t)(hv & 0xFFFF) << 16) | (hv & 0xFFFF);
    uint32_t zw = ((uint32_t)(zp & 0xFFFF) << 16) | (zp & 0xFFFF);
    HVX_Vector vmult = Q6_V_vsplat_R(mw), vhalf = Q6_V_vsplat_R(hw);
    HVX_Vector vzp = Q6_V_vsplat_R(zw), vzero = Q6_V_vzero();

    const uint32_t vt_a0 = VTCM_BASE,        vt_a1 = VTCM_BASE + CH;
    const uint32_t vt_b0 = VTCM_BASE + 2*CH, vt_b1 = VTCM_BASE + 3*CH;
    const uint32_t vt_o0 = VTCM_BASE + 4*CH, vt_o1 = VTCM_BASE + 5*CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) { for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i], b[i], mult, shift, zp); return; }

    d_in_a.next = (uint32_t)(uintptr_t)&d_in_b; d_in_a.ctrl = CH;
    d_in_a.src = (uint32_t)(uintptr_t)a; d_in_a.dst = vt_a0;
    d_in_b.next = 0; d_in_b.ctrl = CH;
    d_in_b.src = (uint32_t)(uintptr_t)b; d_in_b.dst = vt_b0;
    Q6_dmstart_A(&d_in_a); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_a = (c & 1) ? vt_a1 : vt_a0;
        uint32_t cur_b = (c & 1) ? vt_b1 : vt_b0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_a = (c & 1) ? vt_a0 : vt_a1;
        uint32_t nxt_b = (c & 1) ? vt_b0 : vt_b1;

        if (c + 1 < nfull) {
            d_pf_a.next = (uint32_t)(uintptr_t)&d_pf_b; d_pf_a.ctrl = CH;
            d_pf_a.src = (uint32_t)(uintptr_t)(a + (c + 1) * CH); d_pf_a.dst = nxt_a;
            d_pf_b.next = 0; d_pf_b.ctrl = CH;
            d_pf_b.src = (uint32_t)(uintptr_t)(b + (c + 1) * CH); d_pf_b.dst = nxt_b;
            Q6_dmstart_A(&d_pf_a);
        }

        HVX_Vector *pa = (HVX_Vector *)(uintptr_t)cur_a;
        HVX_Vector *pb = (HVX_Vector *)(uintptr_t)cur_b;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int v = 0; v < CH / 128; v++) po[v] = mulrq_vec(pa[v], pb[v], vmult, vhalf, vzp, shift, vzero);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH; d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++) out[i] = ref_scalar(a[i], b[i], mult, shift, zp);
}
