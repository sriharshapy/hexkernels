/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * 2x2 non-overlapping truncated-average pool. DMA-streams row blocks of in[] from
 * DDR into VTCM (double-buffered: next block's DMA overlaps the current block's
 * pool), runs the same HVX pooling on the fast on-chip copies, then DMAs the
 * pooled rows back to DDR. Non-overlapping windows => disjoint input rows per
 * block => no halo. Hides DDR latency on this bandwidth-bound image op.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define BR 16                    /* output rows per tile */

static desc_t d_in, d_pf, d_out;

static void hvx_avgpool2x2(const uint8_t *ip, uint8_t *op, int nrows, int w) {
    int ow = w / 2;
    for (int oy = 0; oy < nrows / 2; oy++) {
        const uint8_t *r0 = ip + (2*oy)   * w;
        const uint8_t *r1 = ip + (2*oy+1) * w;
        uint8_t *o = op + oy * ow;
        int ox = 0;
        for (; ox + 128 <= ow; ox += 128) {
            HVX_Vector r0lo = *(const HVX_Vector *)(r0 + 2*ox);
            HVX_Vector r0hi = *(const HVX_Vector *)(r0 + 2*ox + 128);
            HVX_Vector r1lo = *(const HVX_Vector *)(r1 + 2*ox);
            HVX_Vector r1hi = *(const HVX_Vector *)(r1 + 2*ox + 128);
            HVX_VectorPair d0 = Q6_W_vdeal_VVR(r0hi, r0lo, -1);
            HVX_VectorPair d1 = Q6_W_vdeal_VVR(r1hi, r1lo, -1);
            HVX_VectorPair w0 = Q6_Wh_vadd_VubVub(Q6_V_lo_W(d0), Q6_V_hi_W(d0));
            HVX_VectorPair w1 = Q6_Wh_vadd_VubVub(Q6_V_lo_W(d1), Q6_V_hi_W(d1));
            HVX_Vector slo = Q6_Vh_vadd_VhVh(Q6_V_lo_W(w0), Q6_V_lo_W(w1));
            HVX_Vector shi = Q6_Vh_vadd_VhVh(Q6_V_hi_W(w0), Q6_V_hi_W(w1));
            slo = Q6_Vuh_vlsr_VuhR(slo, 2);
            shi = Q6_Vuh_vlsr_VuhR(shi, 2);
            *(HVX_Vector *)(o + ox) = Q6_Vub_vsat_VhVh(shi, slo);
        }
        for (; ox < ow; ox++) {
            unsigned a=r0[2*ox], b=r0[2*ox+1], c=r1[2*ox], d=r1[2*ox+1];
            o[ox] = (uint8_t)((a+b+c+d)/4);
        }
    }
}

void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h) {
    int ow = w / 2, oh = h / 2;
    const uint32_t CH_IN  = (uint32_t)(2 * BR * w);
    const uint32_t CH_OUT = (uint32_t)(BR * ow);
    const uint32_t vt_i0 = VTCM_BASE,             vt_i1 = VTCM_BASE + CH_IN;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH_IN,   vt_o1 = VTCM_BASE + 2*CH_IN + CH_OUT;

    int ntiles = oh / BR;
    int done_rows = ntiles * BR;

    if (ntiles == 0) { hvx_avgpool2x2(in, out, h, w); return; }

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

        hvx_avgpool2x2((const uint8_t *)(uintptr_t)cur_i, (uint8_t *)(uintptr_t)cur_o, 2 * BR, w);

        if (c + 1 < ntiles) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH_OUT;
        d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * BR * ow);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int oy = done_rows; oy < oh; oy++) {
        const uint8_t *r0 = in + (2*oy)   * w;
        const uint8_t *r1 = in + (2*oy+1) * w;
        for (int ox = 0; ox < ow; ox++) {
            unsigned a=r0[2*ox], b=r0[2*ox+1], c=r1[2*ox], d=r1[2*ox+1];
            out[oy*ow+ox]=(uint8_t)((a+b+c+d)/4);
        }
    }
}
