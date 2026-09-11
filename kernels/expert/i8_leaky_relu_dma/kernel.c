/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams x[] from DDR into VTCM via uDMA, computes int8 leaky-ReLU on the
 * on-chip copy with HVX, DMAs results back. Double-buffered: next input tile is
 * prefetched (async DMA) while the current tile computes, hiding DDR latency. */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384

static desc_t d_in, d_out, d_pf;

static inline HVX_Vector leaky_vec(HVX_Vector vx, HVX_Vector valpha, int shift, HVX_Vector vzero) {
    HVX_VectorPair xh = Q6_Wh_vsxt_Vb(vx);
    HVX_Vector plo = Q6_Vh_vmpyi_VhVh(Q6_V_lo_W(xh), valpha);
    HVX_Vector phi = Q6_Vh_vmpyi_VhVh(Q6_V_hi_W(xh), valpha);
    HVX_Vector sat = Q6_Vb_vasr_VhVhR_sat(phi, plo, shift);
    HVX_VectorPred pos = Q6_Q_vcmp_gt_VbVb(vx, vzero);
    return Q6_V_vmux_QVV(pos, vx, sat);
}

void candidate_kernel(const int8_t *x, int8_t *out, int n, int alpha, int shift) {
    uint32_t aw = ((uint32_t)(alpha & 0xFFFF) << 16) | (alpha & 0xFFFF);
    HVX_Vector valpha = Q6_V_vsplat_R(aw);
    HVX_Vector vzero = Q6_V_vzero();

    const uint32_t vt_x0 = VTCM_BASE,        vt_x1 = VTCM_BASE + CH;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH, vt_o1 = VTCM_BASE + 3*CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) {
            if (x[i] > 0) { out[i] = x[i]; continue; }
            int v = ((int)x[i] * alpha) >> shift;
            if (v > 127) v = 127; if (v < -128) v = -128; out[i] = (int8_t)v;
        }
        return;
    }

    d_in.next = 0; d_in.ctrl = CH; d_in.src = (uint32_t)(uintptr_t)x; d_in.dst = vt_x0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_x = (c & 1) ? vt_x1 : vt_x0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_x = (c & 1) ? vt_x0 : vt_x1;

        if (c + 1 < nfull) {
            const int8_t *nx = x + (c + 1) * CH;
            d_pf.next = 0; d_pf.ctrl = CH; d_pf.src = (uint32_t)(uintptr_t)nx; d_pf.dst = nxt_x;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *px = (HVX_Vector *)(uintptr_t)cur_x;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int v = 0; v < CH / 128; v++) po[v] = leaky_vec(px[v], valpha, shift, vzero);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH; d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++) {
        if (x[i] > 0) { out[i] = x[i]; continue; }
        int v = ((int)x[i] * alpha) >> shift;
        if (v > 127) v = 127; if (v < -128) v = -128; out[i] = (int8_t)v;
    }
}
