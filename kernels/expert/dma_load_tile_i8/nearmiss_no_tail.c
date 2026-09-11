/* Near-miss: correct tile-DMA copy, but forgets the sub-tile remainder
 * (N is not a multiple of CH). The last `rem` bytes stay at their poisoned
 * 0xA5 value -> bit-exact compare fails. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH 16384

static desc_t d_in0, d_pf;

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    const uint32_t vt0 = VTCM_BASE, vt1 = VTCM_BASE + CH;
    int nfull = n / CH;
    if (nfull == 0) return;

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
    /* BUG: no scalar tail loop -> last `rem` bytes of out[] remain poisoned. */
}
