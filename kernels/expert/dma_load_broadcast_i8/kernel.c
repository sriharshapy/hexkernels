/* Solution 1 (= expert): DMA double-buffer a[] through VTCM (ping-pong two
 * slots, prefetch tile c+1 while computing tile c) against the ONCE-loaded
 * broadcast operand vector (op[] is tiny, 128B, loaded straight from DDR a
 * single time -- no need to stage/re-fetch it per tile). The result is
 * computed straight from the on-chip VTCM copy of a[] but STORED DIRECTLY to
 * `out` in DDR via a plain HVX vector store (no output-side DMA round-trip):
 * out[] is written exactly once per element regardless of staging strategy,
 * so routing it through a second VTCM buffer + DMA-out only adds descriptor
 * overhead here with no corresponding DDR-stream consolidation benefit (a[]
 * is the only large stream that needed the bulk-transfer treatment). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 98304

static desc_t d_in0, d_pf;

void candidate_kernel(const int8_t *a, const int8_t *op, int8_t *out, int n) {
    HVX_Vector vop = *(const HVX_Vector *)op;   /* loaded once, reused every chunk */
    const uint32_t vt_a0 = VTCM_BASE, vt_a1 = VTCM_BASE + CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) {
            int32_t t = (int32_t)a[i] + (int32_t)op[i % 128];
            if (t > 127) t = 127; if (t < -128) t = -128;
            out[i] = (int8_t)t;
        }
        return;
    }

    d_in0.next = 0; d_in0.ctrl = CH; d_in0.src = (uint32_t)(uintptr_t)a; d_in0.dst = vt_a0;
    Q6_dmstart_A(&d_in0); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_a = (c & 1) ? vt_a1 : vt_a0;
        uint32_t nxt_a = (c & 1) ? vt_a0 : vt_a1;

        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)(a + (c + 1) * CH); d_pf.dst = nxt_a;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *pa = (HVX_Vector *)(uintptr_t)cur_a;
        HVX_Vector *po = (HVX_Vector *)(out + c * CH);
        for (int v = 0; v < CH / 128; v++) po[v] = Q6_Vb_vadd_VbVb_sat(pa[v], vop);

        if (c + 1 < nfull) Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++) {
        int32_t t = (int32_t)a[i] + (int32_t)op[i % 128];
        if (t > 127) t = 127; if (t < -128) t = -128;
        out[i] = (int8_t)t;
    }
}
