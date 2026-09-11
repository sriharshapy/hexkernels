/* Near-miss: correct DMA-tiled reduction, but forgets the scalar remainder
 * (N is not a multiple of the tile's element count) -- the sum is short by
 * the sum of the last `rem` elements. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH 16384

static desc_t d_in, d_pf;

void candidate_kernel(const int32_t *a, int n, int32_t *out) {
    const uint32_t vt0 = VTCM_BASE, vt1 = VTCM_BASE + CH;
    const int elems_per_tile = CH / (int)sizeof(int32_t);
    int nfull = n / elems_per_tile;
    if (nfull == 0) { out[0] = 0; return; }

    HVX_Vector acc = Q6_V_vzero();

    d_in.next = 0; d_in.ctrl = CH; d_in.src = (uint32_t)(uintptr_t)a; d_in.dst = vt0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur = (c & 1) ? vt1 : vt0;
        uint32_t nxt = (c & 1) ? vt0 : vt1;
        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)(a + (c + 1) * elems_per_tile); d_pf.dst = nxt;
            Q6_dmstart_A(&d_pf);
        }
        HVX_Vector *p = (HVX_Vector *)(uintptr_t)cur;
        for (int v = 0; v < CH / 128; v++) acc = Q6_Vw_vadd_VwVw(acc, p[v]);
        if (c + 1 < nfull) Q6_R_dmwait();
    }

    int32_t lanes[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes = acc;
    int32_t sum = 0;
    for (int j = 0; j < 32; j++) sum += lanes[j];
    /* BUG: no scalar tail accumulation -- last `rem` elements never added. */
    out[0] = sum;
}
