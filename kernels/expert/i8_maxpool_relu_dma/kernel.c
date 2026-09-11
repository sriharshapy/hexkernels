/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Fused 2x2 signed max pool + ReLU. DMA-streams row blocks of in[] from DDR into
 * VTCM (double-buffered: next block's DMA overlaps the current block's pool), runs
 * the same HVX pool+relu on the fast on-chip copies, then DMAs the pooled rows
 * back to DDR. Non-overlapping windows => disjoint input rows per block => no halo.
 * Hides DDR latency on this bandwidth-bound image op.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define BR 16                    /* output rows per tile */

static desc_t d_in, d_pf, d_out;

static void hvx_maxpool_relu(const int8_t *ip, int8_t *op, int nrows, int w) {
    int ow = w / 2;
    HVX_Vector vz = Q6_V_vzero();
    for (int oy = 0; oy < nrows / 2; oy++) {
        const int8_t *r0 = ip + (2*oy)   * w;
        const int8_t *r1 = ip + (2*oy+1) * w;
        int8_t *o = op + oy * ow;
        int x = 0;
        for (; x + 256 <= w; x += 256) {
            HVX_Vector a0 = *(const HVX_Vector *)(r0 + x);
            HVX_Vector a1 = *(const HVX_Vector *)(r0 + x + 128);
            HVX_Vector b0 = *(const HVX_Vector *)(r1 + x);
            HVX_Vector b1 = *(const HVX_Vector *)(r1 + x + 128);
            HVX_Vector m0 = Q6_Vb_vmax_VbVb(a0, b0);
            HVX_Vector m1 = Q6_Vb_vmax_VbVb(a1, b1);
            HVX_VectorPair p = Q6_W_vdeal_VVR(m1, m0, -1);
            HVX_Vector pm = Q6_Vb_vmax_VbVb(Q6_V_lo_W(p), Q6_V_hi_W(p));
            *(HVX_Vector *)(o + x/2) = Q6_Vb_vmax_VbVb(pm, vz);
        }
        for (int ox = x/2; ox < ow; ox++) {
            int a=r0[2*ox], b=r0[2*ox+1], c=r1[2*ox], d=r1[2*ox+1];
            int m=a>b?a:b, n=c>d?c:d; int mm=m>n?m:n; o[ox]=(int8_t)(mm>0?mm:0);
        }
    }
}

void candidate_kernel(const int8_t *in, int8_t *out, int w, int h) {
    int ow = w / 2, oh = h / 2;
    const uint32_t CH_IN  = (uint32_t)(2 * BR * w);
    const uint32_t CH_OUT = (uint32_t)(BR * ow);
    const uint32_t vt_i0 = VTCM_BASE,             vt_i1 = VTCM_BASE + CH_IN;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH_IN,   vt_o1 = VTCM_BASE + 2*CH_IN + CH_OUT;

    int ntiles = oh / BR;
    int done_rows = ntiles * BR;

    if (ntiles == 0) { hvx_maxpool_relu(in, out, h, w); return; }

    d_in.next = 0; d_in.ctrl = CH_IN; d_in.src = (uint32_t)(uintptr_t)in; d_in.dst = vt_i0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < ntiles; c++) {
        uint32_t cur_i = (c & 1) ? vt_i1 : vt_i0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_i = (c & 1) ? vt_i0 : vt_i1;

        if (c + 1 < ntiles) {
            d_pf.next = 0; d_pf.ctrl = CH_IN;
            d_pf.src = (uint32_t)(uintptr_t)(in + (c + 1) * 2 * BR * w); d_pf.dst = nxt_i;
            Q6_dmstart_A(&d_pf);
        }

        hvx_maxpool_relu((const int8_t *)(uintptr_t)cur_i, (int8_t *)(uintptr_t)cur_o, 2 * BR, w);

        if (c + 1 < ntiles) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH_OUT;
        d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * BR * ow);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int oy = done_rows; oy < oh; oy++) {
        const int8_t *r0 = in + (2*oy)   * w;
        const int8_t *r1 = in + (2*oy+1) * w;
        for (int ox = 0; ox < ow; ox++) {
            int a=r0[2*ox], b=r0[2*ox+1], c=r1[2*ox], d=r1[2*ox+1];
            int m=a>b?a:b, n=c>d?c:d; int mm=m>n?m:n; out[oy*ow+ox]=(int8_t)(mm>0?mm:0);
        }
    }
}
