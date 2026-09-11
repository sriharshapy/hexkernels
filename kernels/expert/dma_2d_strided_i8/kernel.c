/* Solution 1 (= expert): bulk-padded DMA + on-chip compaction. Because the
 * rows of a row-group are back-to-back in DDR (each exactly `rowstride`
 * bytes apart), a whole group of RPC padded rows is itself one CONTIGUOUS
 * DDR region -- so it DMAs in with a single flat Type-0 descriptor (no
 * chaining needed, avoiding per-descriptor-link overhead). The padding is
 * then stripped IN VTCM (cheap on-chip HVX copy, no DDR latency): the next
 * group's raw block is prefetched via async DMA while this group's padding
 * is stripped, hiding DDR load latency. The compacted group is then DMA'd
 * out (blocking) with a second flat descriptor. Double buffered on the
 * input side across groups. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define RPC 16                 /* rows per group */

static desc_t d_in0, d_in1, d_out;

void candidate_kernel(const int8_t *a, int8_t *out, int H, int W, int rowstride) {
    const int in_bytes  = RPC * rowstride;
    const int out_bytes = RPC * W;
    const uint32_t vt_in0 = VTCM_BASE;
    const uint32_t vt_in1 = VTCM_BASE + (uint32_t)in_bytes;
    const uint32_t vt_out = VTCM_BASE + (uint32_t)(2 * in_bytes);

    int ngroups  = H / RPC;
    int rem_rows = H - ngroups * RPC;
    const int vlen = sizeof(HVX_Vector);

    if (ngroups == 0) {
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++) out[r * W + c] = a[(long)r * rowstride + c];
        return;
    }

    /* Prologue: DMA group 0's padded block in. */
    d_in0.next = 0; d_in0.ctrl = (uint32_t)in_bytes;
    d_in0.src = (uint32_t)(uintptr_t)a; d_in0.dst = vt_in0;
    Q6_dmstart_A(&d_in0); Q6_R_dmwait();

    for (int g = 0; g < ngroups; g++) {
        uint32_t cur_in = (g & 1) ? vt_in1 : vt_in0;
        uint32_t nxt_in = (g & 1) ? vt_in0 : vt_in1;

        if (g + 1 < ngroups) {
            desc_t *dpf = (g & 1) ? &d_in0 : &d_in1;
            dpf->next = 0; dpf->ctrl = (uint32_t)in_bytes;
            dpf->src = (uint32_t)(uintptr_t)(a + (long)(g + 1) * RPC * rowstride);
            dpf->dst = nxt_in;
            Q6_dmstart_A(dpf);
        }

        /* Strip the padding in VTCM: RPC rows of W bytes, source stride
         * rowstride, dest stride W. Cheap on-chip HVX copy overlaps with
         * the async prefetch above. */
        for (int r = 0; r < RPC; r++) {
            const int8_t *srow = (const int8_t *)(uintptr_t)(cur_in + (uint32_t)(r * rowstride));
            int8_t *drow = (int8_t *)(uintptr_t)(vt_out + (uint32_t)(r * W));
            int c = 0;
            for (; c + vlen <= W; c += vlen)
                *(HVX_Vector *)(drow + c) = *(const HVX_Vector *)(srow + c);
            for (; c < W; c++) drow[c] = srow[c];
        }

        if (g + 1 < ngroups) Q6_R_dmwait();   /* finish the input prefetch */

        d_out.next = 0; d_out.ctrl = (uint32_t)out_bytes;
        d_out.src = vt_out; d_out.dst = (uint32_t)(uintptr_t)(out + (long)g * RPC * W);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int r = ngroups * RPC; r < ngroups * RPC + rem_rows; r++)
        for (int c = 0; c < W; c++) out[r * W + c] = a[(long)r * rowstride + c];
}
