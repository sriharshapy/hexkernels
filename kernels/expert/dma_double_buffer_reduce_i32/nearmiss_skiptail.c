/* Near-miss: correctly DMA-tiles and reduces the FULL-tile portion but
 * forgets the remainder/tail elements entirely (classic "forgot n % chunk"
 * bug). Compiles fine; produces a too-small sum whenever n is not an exact
 * multiple of the DMA chunk size (elems_per_tile=4096) -- which it never is
 * here (300000 % 4096 = 992 != 0). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH 16384

static desc_t d_in, d_pf;

void candidate_kernel(const int32_t *a, int n, int64_t *out) {
    const uint32_t vt0 = VTCM_BASE, vt1 = VTCM_BASE + CH;
    int elems_per_tile = CH / 4;
    int nfull = n / elems_per_tile;
    /* BUG: the remainder (n - nfull*elems_per_tile) elements are silently
     * dropped -- never added to the sum. */

    HVX_Vector acc = Q6_V_vzero();
    if (nfull == 0) { out[0] = 0; return; }

    d_in.next = 0; d_in.ctrl = CH;
    d_in.src = (uint32_t)(uintptr_t)a; d_in.dst = vt0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur = (c & 1) ? vt1 : vt0;
        uint32_t nxt = (c & 1) ? vt0 : vt1;
        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)(a + (c + 1) * elems_per_tile);
            d_pf.dst = nxt;
            Q6_dmstart_A(&d_pf);
        }
        HVX_Vector *p = (HVX_Vector *)(uintptr_t)cur;
        for (int v = 0; v < CH / 128; v++) acc = Q6_Vw_vadd_VwVw(acc, p[v]);
        if (c + 1 < nfull) Q6_R_dmwait();
    }

    int32_t lanes[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes = acc;
    int64_t s = 0;
    for (int j = 0; j < 32; j++) s += (int64_t)lanes[j];
    out[0] = s;   /* tail never added */
}
