/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams a[] from DDR into VTCM via uDMA, computes the saturating int8 abs on
 * the on-chip (fast) copy with HVX, and DMAs results back to DDR.
 * Double-buffered: the next input tile is prefetched (async DMA) while the
 * current tile computes, hiding DDR latency behind compute.
 */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384                 /* bytes per tile (128 vectors) */

static desc_t d_in, d_out, d_pf;

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    const uint32_t vt_x0 = VTCM_BASE,        vt_x1 = VTCM_BASE + CH;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH, vt_o1 = VTCM_BASE + 3*CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) { int v = a[i]; if (v < 0) v = -v; if (v > 127) v = 127; out[i] = (int8_t)v; }
        return;
    }

    d_in.next = 0; d_in.ctrl = CH;
    d_in.src = (uint32_t)(uintptr_t)a; d_in.dst = vt_x0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_x = (c & 1) ? vt_x1 : vt_x0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_x = (c & 1) ? vt_x0 : vt_x1;

        if (c + 1 < nfull) {
            const int8_t *nx = a + (c + 1) * CH;
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)nx; d_pf.dst = nxt_x;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *px = (HVX_Vector *)(uintptr_t)cur_x;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int v = 0; v < CH / 128; v++) po[v] = Q6_Vb_vabs_Vb_sat(px[v]);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH;
        d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++)
        { int v = a[i]; if (v < 0) v = -v; if (v > 127) v = 127; out[i] = (int8_t)v; }
}
