/* Solution 1 (= expert): full double buffer, hvx + dma + vtcm. Ping-pongs
 * two VTCM slots for the input AND two for the output across the N tiles:
 * prefetches tile c+1's input while computing tile c's ReLU, and DMAs tile
 * c's result out immediately after. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 131072

static desc_t d_in0, d_pf, d_out;

void candidate_kernel(const uint8_t *a, uint8_t *out, int n) {
    const uint32_t vt_i0 = VTCM_BASE,        vt_i1 = VTCM_BASE + CH;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH, vt_o1 = VTCM_BASE + 3*CH;
    HVX_Vector bias = Q6_V_vsplat_R(0x80808080);

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) { int v = (int)a[i] - 128; out[i] = (uint8_t)(v < 0 ? 0 : v); }
        return;
    }

    d_in0.next = 0; d_in0.ctrl = CH; d_in0.src = (uint32_t)(uintptr_t)a; d_in0.dst = vt_i0;
    Q6_dmstart_A(&d_in0); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_i = (c & 1) ? vt_i1 : vt_i0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_i = (c & 1) ? vt_i0 : vt_i1;

        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)(a + (c + 1) * CH); d_pf.dst = nxt_i;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *pi = (HVX_Vector *)(uintptr_t)cur_i;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int v = 0; v < CH / 128; v++) po[v] = Q6_Vub_vsub_VubVub_sat(pi[v], bias);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH;
        d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++) {
        int v = (int)a[i] - 128; out[i] = (uint8_t)(v < 0 ? 0 : v);
    }
}
