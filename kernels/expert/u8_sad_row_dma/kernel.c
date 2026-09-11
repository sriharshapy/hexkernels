/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * DMA-streams tiles of BOTH a[] and b[] from DDR into VTCM (double-buffered: the
 * next tile pair's DMA overlaps the current tile's SAD reduce) and runs the
 * sum-of-absolute-differences on the fast on-chip copies. Hides DDR latency
 * behind compute on this bandwidth-bound reduction. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384                 /* bytes per tile per input (128 vectors) */

static desc_t d_in_a, d_in_b, d_pf_a, d_pf_b;

void candidate_kernel(const uint8_t *a, const uint8_t *b, int n, int32_t *out) {
    const uint32_t vt_a0 = VTCM_BASE,        vt_a1 = VTCM_BASE + CH;
    const uint32_t vt_b0 = VTCM_BASE + 2*CH, vt_b1 = VTCM_BASE + 3*CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    HVX_Vector acc  = Q6_V_vzero();
    HVX_Vector ones = Q6_V_vsplat_R(0x01010101);

    if (nfull == 0) {
        int32_t s = 0; for (int i = 0; i < n; i++) { int d=(int)a[i]-(int)b[i]; s += d<0?-d:d; } out[0]=s; return;
    }

    d_in_a.next = (uint32_t)(uintptr_t)&d_in_b; d_in_a.ctrl = CH;
    d_in_a.src = (uint32_t)(uintptr_t)a; d_in_a.dst = vt_a0;
    d_in_b.next = 0; d_in_b.ctrl = CH;
    d_in_b.src = (uint32_t)(uintptr_t)b; d_in_b.dst = vt_b0;
    Q6_dmstart_A(&d_in_a); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_a = (c & 1) ? vt_a1 : vt_a0;
        uint32_t cur_b = (c & 1) ? vt_b1 : vt_b0;
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
        for (int v = 0; v < CH / 128; v++) {
            HVX_Vector d = Q6_Vub_vabsdiff_VubVub(pa[v], pb[v]);
            acc = Q6_Vuw_vrmpyacc_VuwVubVub(acc, d, ones);
        }
        if (c + 1 < nfull) Q6_R_dmwait();
    }

    uint32_t lanes[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes = acc;
    int32_t s = 0;
    for (int j = 0; j < 32; j++) s += (int32_t)lanes[j];
    for (int i = nfull * CH; i < nfull * CH + rem; i++) { int d=(int)a[i]-(int)b[i]; s += d<0?-d:d; }
    out[0] = s;
}
