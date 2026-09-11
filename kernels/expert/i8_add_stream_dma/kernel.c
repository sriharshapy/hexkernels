/* EXPERT (achievability bar) — hvx + dma + vtcm, DOUBLE-BUFFERED, two inputs.
 * Same int8 saturating-add contract as i8_add_dma, but BOTH input streams
 * (a[] and b[]) are prefetched one tile ahead (async uDMA) while the current
 * tile computes, hiding DDR latency on this bandwidth-heavy task (2B read +
 * 1B write per element). Six non-overlapping VTCM slots: a0/a1, b0/b1, o0/o1. */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH_EL 4096
static desc_t d_in_a, d_in_b, d_pf_a, d_pf_b, d_out;

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    const uint32_t vt_a0 = VTCM_BASE,           vt_a1 = VTCM_BASE + CH_EL;
    const uint32_t vt_b0 = VTCM_BASE + 2*CH_EL, vt_b1 = VTCM_BASE + 3*CH_EL;
    const uint32_t vt_o0 = VTCM_BASE + 4*CH_EL, vt_o1 = VTCM_BASE + 5*CH_EL;

    int nfull = n / CH_EL;
    int rem   = n - nfull * CH_EL;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) {
            int s = (int)a[i] + (int)b[i];
            if (s > 127) s = 127;
            if (s < -128) s = -128;
            out[i] = (int8_t)s;
        }
        return;
    }

    /* prime tile 0 for both input streams */
    d_in_a.next = 0; d_in_a.ctrl = CH_EL; d_in_a.src = (uint32_t)(uintptr_t)a; d_in_a.dst = vt_a0;
    Q6_dmstart_A(&d_in_a); Q6_R_dmwait();
    d_in_b.next = 0; d_in_b.ctrl = CH_EL; d_in_b.src = (uint32_t)(uintptr_t)b; d_in_b.dst = vt_b0;
    Q6_dmstart_A(&d_in_b); Q6_R_dmwait();

    const int nbv = CH_EL / 128;                 /* 128B byte-vectors per tile */
    for (int c = 0; c < nfull; c++) {
        uint32_t cur_a = (c & 1) ? vt_a1 : vt_a0;
        uint32_t cur_b = (c & 1) ? vt_b1 : vt_b0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_a = (c & 1) ? vt_a0 : vt_a1;
        uint32_t nxt_b = (c & 1) ? vt_b0 : vt_b1;

        /* CHAIN the two prefetches into ONE descriptor list (a -> b) and issue a
         * single dmstart. Issuing two independent dmstarts back-to-back leaves two
         * chains in flight at once: harmless in an untimed sim, where the first
         * transfer retires instantly, but under timing the second dmstart hits a
         * busy engine and faults (exception 0x28, No Access). One chain, one wait. */
        if (c + 1 < nfull) {
            d_pf_b.next = 0; d_pf_b.ctrl = CH_EL;
            d_pf_b.src = (uint32_t)(uintptr_t)(b + (c + 1) * CH_EL); d_pf_b.dst = nxt_b;
            d_pf_a.next = (uint32_t)(uintptr_t)&d_pf_b; d_pf_a.ctrl = CH_EL;
            d_pf_a.src = (uint32_t)(uintptr_t)(a + (c + 1) * CH_EL); d_pf_a.dst = nxt_a;
            Q6_dmstart_A(&d_pf_a);
        }

        HVX_Vector *pa = (HVX_Vector *)(uintptr_t)cur_a;
        HVX_Vector *pb = (HVX_Vector *)(uintptr_t)cur_b;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int vb = 0; vb < nbv; vb++) po[vb] = Q6_Vb_vadd_VbVb_sat(pa[vb], pb[vb]);

        if (c + 1 < nfull) Q6_R_dmwait();   /* one chain issued -> one wait */

        d_out.next = 0; d_out.ctrl = CH_EL; d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH_EL);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH_EL; i < nfull * CH_EL + rem; i++) {
        int s = (int)a[i] + (int)b[i];
        if (s > 127) s = 127;
        if (s < -128) s = -128;
        out[i] = (int8_t)s;
    }
}
