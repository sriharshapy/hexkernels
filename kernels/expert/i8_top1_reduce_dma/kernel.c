/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Pass 1 (bandwidth-bound): DMA-streams tiles of a[] into VTCM, double-buffered
 * (next tile's DMA overlaps the current tile's vmax reduce), to find the max
 * value V (out[0]) while hiding DDR latency. Pass 2 locates the first index equal
 * to V (out[1]) with an early-exit HVX scan. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384

static desc_t d_in, d_pf;

void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    const int vlen = sizeof(HVX_Vector);
    const uint32_t vt0 = VTCM_BASE, vt1 = VTCM_BASE + CH;
    int nfull = n / CH;
    int rem   = n - nfull * CH;

    HVX_Vector acc = Q6_V_vsplat_R(0x80808080);   /* -128 */

    if (nfull == 0) {
        int V = -128; for (int i = 0; i < n; i++) if ((int)a[i] > V) V = a[i];
        out[0] = V;
        for (int i = 0; i < n; i++) if ((int)a[i] == V) { out[1] = i; return; }
        out[1] = 0; return;
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
        for (int v = 0; v < CH / 128; v++) acc = Q6_Vb_vmax_VbVb(acc, p[v]);
        if (c + 1 < nfull) Q6_R_dmwait();
    }

    int8_t lanes[128] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes = acc;
    int V = -128;
    for (int j = 0; j < 128; j++) if ((int)lanes[j] > V) V = lanes[j];
    for (int i = nfull * CH; i < nfull * CH + rem; i++) if ((int)a[i] > V) V = a[i];
    out[0] = V;

    uint32_t w = ((uint8_t)V) * 0x01010101u;
    HVX_Vector vV = Q6_V_vsplat_R(w);
    int i = 0;
    for (; i + vlen <= n; i += vlen) {
        HVX_VectorPred eq = Q6_Q_vcmp_eq_VbVb(*(const HVX_Vector *)(a + i), vV);
        int8_t tmp[128] __attribute__((aligned(128)));
        *(HVX_Vector *)tmp = Q6_V_vand_QR(eq, -1);
        for (int j = 0; j < vlen; j++) if ((uint8_t)tmp[j] == 0xFF) { out[1] = i + j; return; }
    }
    for (; i < n; i++) if ((int)a[i] == V) { out[1] = i; return; }
    out[1] = 0;
}
