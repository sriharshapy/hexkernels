/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * DMA-streams tiles of BOTH a[] and b[] (int16) from DDR into VTCM (double-
 * buffered: next tile pair's DMA overlaps the current tile's multiply-accumulate)
 * and runs the int16 dot-product reduction on the fast on-chip copies. Hides DDR
 * latency behind compute on this bandwidth-bound reduction. Tiles are addressed
 * in BYTES (CH bytes per input tile). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384                 /* bytes per tile per input (128 vectors, 8192 int16) */

static desc_t d_in_a, d_in_b, d_pf_a, d_pf_b;

void candidate_kernel(const int16_t *a, const int16_t *b, int n, int32_t *out) {
    const uint32_t vt_a0 = VTCM_BASE,        vt_a1 = VTCM_BASE + CH;
    const uint32_t vt_b0 = VTCM_BASE + 2*CH, vt_b1 = VTCM_BASE + 3*CH;

    const uint8_t *ab = (const uint8_t *)a;
    const uint8_t *bb = (const uint8_t *)b;
    int nbytes = n * 2;
    int nfull  = nbytes / CH;
    int rembytes = nbytes - nfull * CH;

    HVX_Vector acc_lo = Q6_V_vzero(), acc_hi = Q6_V_vzero();

    if (nfull == 0) {
        int32_t s = 0; for (int i = 0; i < n; i++) s += (int32_t)a[i]*(int32_t)b[i]; out[0] = s; return;
    }

    d_in_a.next = (uint32_t)(uintptr_t)&d_in_b; d_in_a.ctrl = CH;
    d_in_a.src = (uint32_t)(uintptr_t)ab; d_in_a.dst = vt_a0;
    d_in_b.next = 0; d_in_b.ctrl = CH;
    d_in_b.src = (uint32_t)(uintptr_t)bb; d_in_b.dst = vt_b0;
    Q6_dmstart_A(&d_in_a); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_a = (c & 1) ? vt_a1 : vt_a0;
        uint32_t cur_b = (c & 1) ? vt_b1 : vt_b0;
        uint32_t nxt_a = (c & 1) ? vt_a0 : vt_a1;
        uint32_t nxt_b = (c & 1) ? vt_b0 : vt_b1;

        if (c + 1 < nfull) {
            d_pf_a.next = (uint32_t)(uintptr_t)&d_pf_b; d_pf_a.ctrl = CH;
            d_pf_a.src = (uint32_t)(uintptr_t)(ab + (c + 1) * CH); d_pf_a.dst = nxt_a;
            d_pf_b.next = 0; d_pf_b.ctrl = CH;
            d_pf_b.src = (uint32_t)(uintptr_t)(bb + (c + 1) * CH); d_pf_b.dst = nxt_b;
            Q6_dmstart_A(&d_pf_a);
        }

        HVX_Vector *pa = (HVX_Vector *)(uintptr_t)cur_a;
        HVX_Vector *pb = (HVX_Vector *)(uintptr_t)cur_b;
        for (int v = 0; v < CH / 128; v++) {
            HVX_VectorPair p = Q6_Ww_vmpy_VhVh(pa[v], pb[v]);
            acc_lo = Q6_Vw_vadd_VwVw(acc_lo, Q6_V_lo_W(p));
            acc_hi = Q6_Vw_vadd_VwVw(acc_hi, Q6_V_hi_W(p));
        }
        if (c + 1 < nfull) Q6_R_dmwait();
    }

    int32_t lo[32] __attribute__((aligned(128))), hi[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lo = acc_lo; *(HVX_Vector *)hi = acc_hi;
    int32_t s = 0;
    for (int j = 0; j < 32; j++) s += lo[j] + hi[j];
    for (int i = nfull * (CH / 2); i < nfull * (CH / 2) + rembytes / 2; i++)
        s += (int32_t)a[i] * (int32_t)b[i];
    out[0] = s;
}
