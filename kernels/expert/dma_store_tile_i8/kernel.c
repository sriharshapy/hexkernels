/* Solution 1 (= expert): double-buffered DMA-store. Reads a[] tile directly
 * from DDR with HVX, computes bitwise-NOT into VTCM slot c, and starts an
 * async DMA-store of slot c while computing slot c+1's result; the DMA-out
 * of slot c-1 finishes overlapped with slot c's compute. */
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
    int rem   = n - nfull * CH;

    int pending = 0;   /* is there an in-flight store from the previous iter? */

    for (int c = 0; c < nfull; c++) {
        uint32_t cur = (c & 1) ? vt1 : vt0;

        const HVX_Vector *pa = (const HVX_Vector *)(a + c * CH);
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur;
        for (int v = 0; v < CH / 128; v++) po[v] = Q6_V_vnot_V(pa[v]);

        if (pending) Q6_R_dmwait();   /* finish the previous tile's store */

        d_out.next = 0; d_out.ctrl = CH;
        d_out.src = cur; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out);
        pending = 1;
    }
    if (pending) Q6_R_dmwait();

    for (int i = nfull * CH; i < nfull * CH + rem; i++) out[i] = (int8_t)(~a[i]);
}
