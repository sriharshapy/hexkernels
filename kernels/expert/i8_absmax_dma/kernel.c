/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * DMA-streams tiles of a[] from DDR into VTCM (double-buffered) and runs the
 * absolute-max reduction on the fast on-chip copies. Bytes are sign-extended to
 * halfwords so |-128|=128 is exact; halfword abs + vmax accumulate, then the 64
 * lanes are horizontally maxed. Hides DDR latency behind compute. */
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

    if (nfull == 0) {
        int32_t m = 0; for (int i = 0; i < n; i++) { int32_t v = a[i]<0?-(int32_t)a[i]:(int32_t)a[i]; if (v>m) m=v; } out[0]=m; return;
    }

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
        for (int v = 0; v < CH / 128; v++) {
            HVX_VectorPair w = Q6_Wh_vsxt_Vb(p[v]);
            HVX_Vector lo = Q6_Vh_vabs_Vh(Q6_V_lo_W(w));
            HVX_Vector hi = Q6_Vh_vabs_Vh(Q6_V_hi_W(w));
            acc = Q6_Vh_vmax_VhVh(acc, Q6_Vh_vmax_VhVh(lo, hi));
        }
        if (c + 1 < nfull) Q6_R_dmwait();
    }

    int16_t lanes[64] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes = acc;
    int32_t m = 0;
    for (int j = 0; j < 64; j++) if (lanes[j] > m) m = lanes[j];
    for (int i = nfull * CH; i < nfull * CH + rem; i++) { int32_t v = a[i]<0?-(int32_t)a[i]:(int32_t)a[i]; if (v>m) m=v; }
    out[0] = m;
}
