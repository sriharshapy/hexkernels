/* EXPERT (achievability bar) = solutions/s1.c: double-buffered DMA-load.
 * Tile c+1 is prefetched via uDMA into VTCM while tile c (already resident)
 * is written out with a plain HVX vector store, hiding DDR load latency
 * behind the store loop. */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384                 /* bytes per tile (128 vectors) */

static desc_t d_in0, d_pf;

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    const uint32_t vt0 = VTCM_BASE, vt1 = VTCM_BASE + CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) { for (int i = 0; i < n; i++) out[i] = a[i]; return; }

    /* Prologue: DMA tile 0 into slot 0. */
    d_in0.next = 0; d_in0.ctrl = CH; d_in0.src = (uint32_t)(uintptr_t)a; d_in0.dst = vt0;
    Q6_dmstart_A(&d_in0); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur = (c & 1) ? vt1 : vt0;
        uint32_t nxt = (c & 1) ? vt0 : vt1;

        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)(a + (c + 1) * CH); d_pf.dst = nxt;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *p = (HVX_Vector *)(uintptr_t)cur;
        HVX_Vector *po = (HVX_Vector *)(out + c * CH);
        for (int v = 0; v < CH / 128; v++) po[v] = p[v];

        if (c + 1 < nfull) Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++) out[i] = a[i];
}
