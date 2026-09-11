/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * DMA-streams tiles of a[] from DDR into VTCM (double-buffered: next tile's DMA
 * overlaps current tile's reduce) and runs the vrmpy reduction on the fast
 * on-chip copies. Hides DDR latency behind compute on this bandwidth-bound
 * reduction.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384                 /* bytes per tile (128 vectors) */

static desc_t d_in, d_pf;

void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    const uint32_t vt0 = VTCM_BASE, vt1 = VTCM_BASE + CH;
    int nfull = n / CH;
    int rem   = n - nfull * CH;

    HVX_Vector acc = Q6_V_vzero();
    HVX_Vector ones = Q6_V_vsplat_R(0x01010101);

    if (nfull == 0) {
        int32_t s = 0; for (int i = 0; i < n; i++) s += (int32_t)a[i]; out[0] = s; return;
    }

    /* Prologue: DMA tile 0 into slot 0. */
    d_in.next = 0; d_in.ctrl = CH; d_in.src = (uint32_t)(uintptr_t)a; d_in.dst = vt0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur = (c & 1) ? vt1 : vt0;
        uint32_t nxt = (c & 1) ? vt0 : vt1;
        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)(a + (c + 1) * CH); d_pf.dst = nxt;
            Q6_dmstart_A(&d_pf);
        }
        HVX_Vector *p = (HVX_Vector *)(uintptr_t)cur;
        for (int v = 0; v < CH / 128; v++)
            acc = Q6_Vw_vrmpyacc_VwVbVb(acc, p[v], ones);
        if (c + 1 < nfull) Q6_R_dmwait();
    }

    int32_t lanes[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes = acc;
    int32_t s = 0;
    for (int j = 0; j < 32; j++) s += lanes[j];
    for (int i = nfull * CH; i < nfull * CH + rem; i++) s += (int32_t)a[i];
    out[0] = s;
}
