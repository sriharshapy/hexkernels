/* EXPERT (achievability bar) — hvx + dma + vtcm, DOUBLE-BUFFERED.
 * Same int8 ReLU contract as i8_relu_dma, but the next input tile is
 * prefetched (async uDMA) while the current tile computes, hiding DDR
 * latency on this bandwidth-heavy task (1B read + 1B write per element). */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH_EL 4096
static desc_t d_in, d_out, d_pf;

void candidate_kernel(const int8_t *x, int8_t *out, int n) {
    HVX_Vector vzero = Q6_V_vzero();

    const uint32_t vt_x0 = VTCM_BASE,           vt_x1 = VTCM_BASE + CH_EL;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH_EL, vt_o1 = VTCM_BASE + 3*CH_EL;

    int nfull = n / CH_EL;
    int rem   = n - nfull * CH_EL;

    if (nfull == 0) { for (int i = 0; i < n; i++) out[i] = (x[i] < 0) ? 0 : x[i]; return; }

    d_in.next = 0; d_in.ctrl = CH_EL; d_in.src = (uint32_t)(uintptr_t)x; d_in.dst = vt_x0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    const int nbv = CH_EL / 128;                 /* 128B byte-vectors per tile */
    for (int c = 0; c < nfull; c++) {
        uint32_t cur_x = (c & 1) ? vt_x1 : vt_x0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_x = (c & 1) ? vt_x0 : vt_x1;

        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH_EL;
            d_pf.src = (uint32_t)(uintptr_t)(x + (c + 1) * CH_EL); d_pf.dst = nxt_x;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *px = (HVX_Vector *)(uintptr_t)cur_x;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int vb = 0; vb < nbv; vb++) po[vb] = Q6_Vb_vmax_VbVb(px[vb], vzero);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH_EL; d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH_EL);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH_EL; i < nfull * CH_EL + rem; i++) out[i] = (x[i] < 0) ? 0 : x[i];
}
