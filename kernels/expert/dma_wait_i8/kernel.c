/* Solution 1 (= expert): double-buffered DMA-load, synchronized by
 * *polling* Q6_R_dmpoll() in a spin loop rather than blocking on the
 * dedicated Q6_R_dmwait() instruction. Tile c+1 is prefetched via uDMA into
 * VTCM while tile c's saturating add-50 is computed and stored to out with
 * a plain HVX vector store. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384

static desc_t d_in0, d_pf;

static inline void dma_spin_wait(void) {
    while (Q6_R_dmpoll() != 0) { /* spin until the DMA queue reports idle */ }
}

void candidate_kernel(const uint8_t *a, uint8_t *out, int n) {
    const uint32_t vt0 = VTCM_BASE, vt1 = VTCM_BASE + CH;
    HVX_Vector fifty = Q6_V_vsplat_R(0x32323232);

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) { int v = (int)a[i] + 50; out[i] = (uint8_t)(v > 255 ? 255 : v); }
        return;
    }

    d_in0.next = 0; d_in0.ctrl = CH; d_in0.src = (uint32_t)(uintptr_t)a; d_in0.dst = vt0;
    Q6_dmstart_A(&d_in0);
    dma_spin_wait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur = (c & 1) ? vt1 : vt0;
        uint32_t nxt = (c & 1) ? vt0 : vt1;

        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)(a + (c + 1) * CH); d_pf.dst = nxt;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *pi = (HVX_Vector *)(uintptr_t)cur;
        HVX_Vector *po = (HVX_Vector *)(out + c * CH);
        for (int v = 0; v < CH / 128; v++) po[v] = Q6_Vub_vadd_VubVub_sat(pi[v], fifty);

        if (c + 1 < nfull) dma_spin_wait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++) {
        int v = (int)a[i] + 50; out[i] = (uint8_t)(v > 255 ? 255 : v);
    }
}
