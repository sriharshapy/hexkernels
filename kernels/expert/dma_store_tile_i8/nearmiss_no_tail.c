/* Near-miss: correct double-buffered DMA-store, but forgets the sub-tile
 * remainder (N is not a multiple of CH). The last `rem` bytes of out[] stay
 * at their poisoned 0xA5 value -> bit-exact compare fails. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH 16384

static desc_t d_out;

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    const uint32_t vt0 = VTCM_BASE, vt1 = VTCM_BASE + CH;
    int nfull = n / CH;
    int pending = 0;

    for (int c = 0; c < nfull; c++) {
        uint32_t cur = (c & 1) ? vt1 : vt0;
        const HVX_Vector *pa = (const HVX_Vector *)(a + c * CH);
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur;
        for (int v = 0; v < CH / 128; v++) po[v] = Q6_V_vnot_V(pa[v]);

        if (pending) Q6_R_dmwait();
        d_out.next = 0; d_out.ctrl = CH;
        d_out.src = cur; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out);
        pending = 1;
    }
    if (pending) Q6_R_dmwait();
    /* BUG: no scalar tail loop -> last `rem` bytes of out[] remain poisoned. */
}
