/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams int8 x[] from DDR into VTCM via uDMA, computes int8 hard-swish on the
 * on-chip copy with HVX, and DMAs results back. Double-buffered: the next input
 * tile is prefetched (async DMA) while the current tile computes, hiding DDR
 * latency on this bandwidth-bound task (1B read/1B write per element).
 * x*relu6 in halfword lanes; /6 truncate-toward-zero via word-lane magic. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH 16384                 /* bytes per tile (128 vectors) */
static desc_t d_in, d_out, d_pf;

#define HSW_MAG ((10923 & 0xFFFF) | ((10923 & 0xFFFF) << 16))

static inline HVX_Vector hswish_half(HVX_Vector hv, HVX_Vector v3, HVX_Vector v0h,
                                     HVX_Vector v6, HVX_Vector vlo, HVX_Vector vhi,
                                     HVX_Vector vzeroh) {
    HVX_Vector r6 = Q6_Vh_vadd_VhVh(hv, v3);
    r6 = Q6_Vh_vmax_VhVh(r6, v0h);
    r6 = Q6_Vh_vmin_VhVh(r6, v6);
    HVX_Vector prod = Q6_Vh_vmpyi_VhVh(hv, r6);
    HVX_Vector absp = Q6_Vh_vabs_Vh(prod);
    HVX_VectorPair apw = Q6_Ww_vsxt_Vh(absp);
    HVX_Vector qlo = Q6_Vw_vasr_VwR(Q6_Vw_vmpyi_VwRh(Q6_V_lo_W(apw), HSW_MAG), 16);
    HVX_Vector qhi = Q6_Vw_vasr_VwR(Q6_Vw_vmpyi_VwRh(Q6_V_hi_W(apw), HSW_MAG), 16);
    HVX_Vector q16 = Q6_Vh_vasr_VwVwR_sat(qhi, qlo, 0);
    HVX_Vector negq = Q6_Vh_vsub_VhVh(vzeroh, q16);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzeroh, prod);
    HVX_Vector res = Q6_V_vmux_QVV(neg, negq, q16);
    res = Q6_Vh_vmax_VhVh(res, vlo);
    res = Q6_Vh_vmin_VhVh(res, vhi);
    return res;
}
static inline HVX_Vector hswish_bytevec(HVX_Vector xb, HVX_Vector v3, HVX_Vector v0h,
                                        HVX_Vector v6, HVX_Vector vlo, HVX_Vector vhi,
                                        HVX_Vector vzeroh) {
    HVX_VectorPair xh = Q6_Wh_vsxt_Vb(xb);
    HVX_Vector rlo = hswish_half(Q6_V_lo_W(xh), v3, v0h, v6, vlo, vhi, vzeroh);
    HVX_Vector rhi = hswish_half(Q6_V_hi_W(xh), v3, v0h, v6, vlo, vhi, vzeroh);
    return Q6_Vb_vasr_VhVhR_sat(rhi, rlo, 0);
}
static int8_t ref_scalar(int8_t xi) {
    int v = (int)xi, r6 = v + 3; if (r6 < 0) r6 = 0; else if (r6 > 6) r6 = 6;
    int result = (v * r6) / 6; if (result > 127) result = 127; if (result < -128) result = -128;
    return (int8_t)result;
}

void candidate_kernel(const int8_t *x, int8_t *out, int n) {
    HVX_Vector v3    = Q6_V_vsplat_R(0x00030003u);
    HVX_Vector v0h   = Q6_V_vzero();
    HVX_Vector v6    = Q6_V_vsplat_R(0x00060006u);
    HVX_Vector vlo   = Q6_V_vsplat_R(0xFF80FF80u);
    HVX_Vector vhi   = Q6_V_vsplat_R(0x007F007Fu);
    HVX_Vector vzeroh= Q6_V_vzero();

    const uint32_t vt_x0 = VTCM_BASE,        vt_x1 = VTCM_BASE + CH;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH, vt_o1 = VTCM_BASE + 3*CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) { for (int i = 0; i < n; i++) out[i] = ref_scalar(x[i]); return; }

    d_in.next = 0; d_in.ctrl = CH; d_in.src = (uint32_t)(uintptr_t)x; d_in.dst = vt_x0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_x = (c & 1) ? vt_x1 : vt_x0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_x = (c & 1) ? vt_x0 : vt_x1;

        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)(x + (c + 1) * CH); d_pf.dst = nxt_x;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *px = (HVX_Vector *)(uintptr_t)cur_x;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int v = 0; v < CH / 128; v++)
            po[v] = hswish_bytevec(px[v], v3, v0h, v6, vlo, vhi, vzeroh);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH; d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++) out[i] = ref_scalar(x[i]);
}
