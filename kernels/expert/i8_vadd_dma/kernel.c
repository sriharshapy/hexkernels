/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams a[] and b[] from DDR into VTCM via uDMA, computes the int8 add on the
 * on-chip (fast) copies with HVX, and DMAs results back to DDR. Double-buffered:
 * the next input tile is prefetched (async DMA) while the current tile computes,
 * hiding DDR latency behind compute. VTCM residence + overlap beats the plain
 * DDR HVX loop on this bandwidth-bound task.
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

static desc_t d_in_a, d_in_b, d_out, d_pf_a, d_pf_b;

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    const uint32_t vt_a0 = VTCM_BASE,        vt_a1 = VTCM_BASE + CH;
    const uint32_t vt_b0 = VTCM_BASE + 2*CH, vt_b1 = VTCM_BASE + 3*CH;
    const uint32_t vt_o0 = VTCM_BASE + 4*CH, vt_o1 = VTCM_BASE + 5*CH;

    int nfull = n / CH;                 /* number of full CH-byte tiles */
    int rem   = n - nfull * CH;

    if (nfull == 0) { /* tiny n: scalar fallback */
        for (int i = 0; i < n; i++) out[i] = (int8_t)(a[i] + b[i]);
        return;
    }

    /* Prologue: DMA tile 0 (a and b) into slot 0, chained descriptor. */
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

        /* Prefetch next input tile (async) while we compute the current one. */
        if (c + 1 < nfull) {
            const int8_t *na = a + (c + 1) * CH;
            const int8_t *nb = b + (c + 1) * CH;
            d_pf_a.next = (uint32_t)(uintptr_t)&d_pf_b; d_pf_a.ctrl = CH;
            d_pf_a.src = (uint32_t)(uintptr_t)na; d_pf_a.dst = nxt_a;
            d_pf_b.next = 0; d_pf_b.ctrl = CH;
            d_pf_b.src = (uint32_t)(uintptr_t)nb; d_pf_b.dst = nxt_b;
            Q6_dmstart_A(&d_pf_a);
        }

        /* Compute add on the on-chip tile. */
        HVX_Vector *pa = (HVX_Vector *)(uintptr_t)cur_a;
        HVX_Vector *pb = (HVX_Vector *)(uintptr_t)cur_b;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int v = 0; v < CH / 128; v++) po[v] = Q6_Vb_vadd_VbVb(pa[v], pb[v]);

        if (c + 1 < nfull) Q6_R_dmwait();   /* finish prefetch */

        /* Store this output tile back to DDR (blocking). */
        d_out.next = 0; d_out.ctrl = CH;
        d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    /* Scalar tail for the sub-tile remainder. */
    for (int i = nfull * CH; i < nfull * CH + rem; i++)
        out[i] = (int8_t)(a[i] + b[i]);
}
