/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams in[] from DDR into VTCM via uDMA, computes the inclusive in-range test
 * [64,191] on the on-chip copy with HVX (ge_lo AND NOT gt_hi), and DMAs results
 * back to DDR. Double-buffered: the next input tile is prefetched (async DMA)
 * while the current tile computes, hiding DDR latency behind compute.
 */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

/* uDMA Type-0 (1D) descriptor, 16 bytes. MUST be static/global (a stack
 * descriptor silently no-ops at -O2). */
typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384                 /* bytes per tile (128 vectors) */
#define LO 64
#define HI 191

static desc_t d_in, d_out, d_pf;

void candidate_kernel(const uint8_t *in, uint8_t *out, int n) {
    const uint32_t vt_x0 = VTCM_BASE,        vt_x1 = VTCM_BASE + CH;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH, vt_o1 = VTCM_BASE + 3*CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) out[i] = (uint8_t)((in[i] >= LO && in[i] <= HI) ? 255 : 0);
        return;
    }

    d_in.next = 0; d_in.ctrl = CH;
    d_in.src = (uint32_t)(uintptr_t)in; d_in.dst = vt_x0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    HVX_Vector vlo1 = Q6_Vb_vsplat_R(LO - 1);   /* 63  */
    HVX_Vector vhi  = Q6_Vb_vsplat_R(HI);       /* 191 */

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_x = (c & 1) ? vt_x1 : vt_x0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_x = (c & 1) ? vt_x0 : vt_x1;

        if (c + 1 < nfull) {
            const uint8_t *nx = in + (c + 1) * CH;
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)nx; d_pf.dst = nxt_x;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *px = (HVX_Vector *)(uintptr_t)cur_x;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int v = 0; v < CH / 128; v++) {
            HVX_Vector x = px[v];
            HVX_VectorPred p_ge_lo = Q6_Q_vcmp_gt_VubVub(x, vlo1);
            HVX_VectorPred p_gt_hi = Q6_Q_vcmp_gt_VubVub(x, vhi);
            HVX_VectorPred p = Q6_Q_and_QQn(p_ge_lo, p_gt_hi);
            po[v] = Q6_V_vand_QR(p, -1);
        }

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH;
        d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++)
        out[i] = (uint8_t)((in[i] >= LO && in[i] <= HI) ? 255 : 0);
}
