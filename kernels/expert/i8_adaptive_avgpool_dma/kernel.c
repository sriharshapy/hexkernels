/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Each pooling window is exactly one CH=16384-byte tile (k = n/m = 16384).
 * DMA-streams each window from DDR into VTCM (double-buffered: next window's DMA
 * overlaps the current window's reduce) and runs the unsigned vrmpy reduction on
 * the fast on-chip copy, emitting one int8 mean per window. Hides DDR latency
 * behind compute on this bandwidth-bound adaptive average pool. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384                 /* bytes per tile = window size k */

static desc_t d_in, d_pf;

void candidate_kernel(const uint8_t *in, int8_t *out, int n, int m) {
    int k = n / m;
    const uint32_t vt0 = VTCM_BASE, vt1 = VTCM_BASE + CH;
    HVX_Vector ones = Q6_V_vsplat_R(0x01010101);

    /* Fallback if the window size is not the expected full tile. */
    if (k != CH) {
        for (int i = 0; i < m; i++) {
            const uint8_t *w = in + (long)i * k;
            int32_t s = 0; for (int j = 0; j < k; j++) s += (int32_t)w[j];
            out[i] = (int8_t)(s / k);
        }
        return;
    }

    /* Prologue: DMA window 0 into slot 0. */
    d_in.next = 0; d_in.ctrl = CH; d_in.src = (uint32_t)(uintptr_t)in; d_in.dst = vt0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < m; c++) {
        uint32_t cur = (c & 1) ? vt1 : vt0;
        uint32_t nxt = (c & 1) ? vt0 : vt1;
        if (c + 1 < m) {
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)(in + (long)(c + 1) * CH); d_pf.dst = nxt;
            Q6_dmstart_A(&d_pf);
        }
        HVX_Vector *p = (HVX_Vector *)(uintptr_t)cur;
        HVX_Vector acc = Q6_V_vzero();
        for (int v = 0; v < CH / 128; v++)
            acc = Q6_Vuw_vrmpyacc_VuwVubVub(acc, p[v], ones);
        uint32_t lanes[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = acc;
        int32_t s = 0;
        for (int t = 0; t < 32; t++) s += (int32_t)lanes[t];
        out[c] = (int8_t)(s / k);
        if (c + 1 < m) Q6_R_dmwait();
    }
}
