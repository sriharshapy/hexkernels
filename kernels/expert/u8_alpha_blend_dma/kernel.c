/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams a[] and b[] from DDR into VTCM via uDMA, computes the uint8 alpha blend
 * on the on-chip copies with HVX, and DMAs results back to DDR. Double-buffered:
 * the next input tiles are prefetched (async DMA) while the current tile computes,
 * hiding DDR latency on this bandwidth-bound task. Halfword-lane blend via
 * out = b + ((alpha*(a-b)+128)>>8). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH 16384
static desc_t d_in_a, d_in_b, d_out, d_pf_a, d_pf_b;

static inline HVX_Vector blend_vec(HVX_Vector va, HVX_Vector vb, HVX_Vector valpha,
                                   HVX_Vector v128) {
    HVX_VectorPair ah = Q6_Wuh_vzxt_Vub(va);
    HVX_VectorPair bh = Q6_Wuh_vzxt_Vub(vb);
    HVX_Vector al[2] = { Q6_V_lo_W(ah), Q6_V_hi_W(ah) };
    HVX_Vector bl[2] = { Q6_V_lo_W(bh), Q6_V_hi_W(bh) };
    HVX_Vector res[2];
    for (int k = 0; k < 2; k++) {
        HVX_Vector t = Q6_Vh_vsub_VhVh(al[k], bl[k]);
        HVX_Vector X = Q6_Vh_vmpyi_VhVh(t, valpha);
        HVX_Vector y = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(X, v128), 8);
        res[k] = Q6_Vh_vadd_VhVh(y, bl[k]);
    }
    return Q6_Vub_vasr_VhVhR_sat(res[1], res[0], 0);
}

void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out,
                      int n, uint16_t alpha) {
    uint32_t aw = ((uint32_t)alpha << 16) | (uint32_t)alpha;
    HVX_Vector valpha = Q6_V_vsplat_R(aw);
    HVX_Vector v128   = Q6_V_vsplat_R((128u << 16) | 128u);

    const uint32_t vt_a0 = VTCM_BASE,        vt_a1 = VTCM_BASE + CH;
    const uint32_t vt_b0 = VTCM_BASE + 2*CH, vt_b1 = VTCM_BASE + 3*CH;
    const uint32_t vt_o0 = VTCM_BASE + 4*CH, vt_o1 = VTCM_BASE + 5*CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) {
            int v = ((int)alpha*(int)a[i] + (256-(int)alpha)*(int)b[i] + 128) >> 8;
            out[i] = (uint8_t)v;
        }
        return;
    }

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
        for (int v = 0; v < CH / 128; v++) po[v] = blend_vec(pa[v], pb[v], valpha, v128);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH; d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++) {
        int v = ((int)alpha*(int)a[i] + (256-(int)alpha)*(int)b[i] + 128) >> 8;
        out[i] = (uint8_t)v;
    }
}
