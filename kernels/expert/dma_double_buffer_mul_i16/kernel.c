/* Solution 1 (= expert): DMA double-buffer, hvx + dma + vtcm, int16 vector
 * width. Ping-pongs two VTCM slots for a, b, and out across the N tiles:
 * prefetches tile c+1 while computing/storing tile c. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384   /* bytes per tile = 8192 int16 elements */

static desc_t d_in_a, d_in_b, d_out, d_pf_a, d_pf_b;

void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    const uint32_t vt_a0 = VTCM_BASE,        vt_a1 = VTCM_BASE + CH;
    const uint32_t vt_b0 = VTCM_BASE + 2*CH, vt_b1 = VTCM_BASE + 3*CH;
    const uint32_t vt_o0 = VTCM_BASE + 4*CH, vt_o1 = VTCM_BASE + 5*CH;
    const int elems_per_tile = CH / (int)sizeof(int16_t);

    int nfull = n / elems_per_tile;
    int rem   = n - nfull * elems_per_tile;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) out[i] = (int16_t)((int32_t)a[i] * (int32_t)b[i]);
        return;
    }

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

        if (c + 1 < nfull) {
            const int16_t *na = a + (c + 1) * elems_per_tile;
            const int16_t *nb = b + (c + 1) * elems_per_tile;
            d_pf_a.next = (uint32_t)(uintptr_t)&d_pf_b; d_pf_a.ctrl = CH;
            d_pf_a.src = (uint32_t)(uintptr_t)na; d_pf_a.dst = nxt_a;
            d_pf_b.next = 0; d_pf_b.ctrl = CH;
            d_pf_b.src = (uint32_t)(uintptr_t)nb; d_pf_b.dst = nxt_b;
            Q6_dmstart_A(&d_pf_a);
        }

        HVX_Vector *pa = (HVX_Vector *)(uintptr_t)cur_a;
        HVX_Vector *pb = (HVX_Vector *)(uintptr_t)cur_b;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int v = 0; v < CH / 128; v++) po[v] = Q6_Vh_vmpyi_VhVh(pa[v], pb[v]);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH;
        d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * elems_per_tile);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * elems_per_tile; i < nfull * elems_per_tile + rem; i++)
        out[i] = (int16_t)((int32_t)a[i] * (int32_t)b[i]);
}
