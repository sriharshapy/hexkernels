/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Fused 2x2 avg-pool (signed, trunc toward zero) + requant. DMA-streams row blocks
 * of in[] from DDR into VTCM (double-buffered: next block's DMA overlaps the current
 * block's compute), runs the same HVX pool+requant on the fast on-chip copies, then
 * DMAs the pooled/requantized rows back to DDR. Non-overlapping windows => disjoint
 * input rows per block => no halo. Hides DDR latency on this bandwidth-bound op.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define BR 16                    /* output rows per tile */

static desc_t d_in, d_pf, d_out;

static inline HVX_Vector requant_h(HVX_Vector pool, HVX_Vector vmult, HVX_Vector vhalf,
                                   int shift, HVX_Vector vzp, HVX_Vector vzero) {
    HVX_Vector v    = Q6_Vh_vmpyi_VhVh(pool, vmult);
    HVX_Vector absv = Q6_Vh_vabs_Vh(v);
    HVX_Vector sh   = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absv, vhalf), shift);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);
    HVX_Vector r    = Q6_V_vmux_QVV(neg, Q6_Vh_vsub_VhVh(vzero, sh), sh);
    return Q6_Vh_vadd_VhVh(r, vzp);
}

static int8_t ref_element(int pool, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)pool * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp; if (r > 127) r = 127; if (r < -128) r = -128; return (int8_t)r;
}

static void hvx_avgpool_requant(const int8_t *ip, int8_t *op, int w, int nrows,
                                int32_t mult, int shift, int8_t zp) {
    int ow = w / 2;
    HVX_Vector vzero = Q6_V_vzero();
    HVX_Vector vmult = Q6_Vh_vsplat_R(mult);
    int hv = shift > 0 ? (1 << (shift - 1)) : 0;
    HVX_Vector vhalf = Q6_Vh_vsplat_R(hv);
    HVX_Vector vzp   = Q6_Vh_vsplat_R((int)zp);
    for (int oy = 0; oy < nrows / 2; oy++) {
        const int8_t *r0 = ip + (2*oy)   * w;
        const int8_t *r1 = ip + (2*oy+1) * w;
        int8_t *o = op + oy * ow;
        int X = 0;
        for (; X + 256 <= w; X += 256) {
            HVX_VectorPair d0 = Q6_W_vdeal_VVR(*(const HVX_Vector *)(r0 + X + 128), *(const HVX_Vector *)(r0 + X), -1);
            HVX_VectorPair d1 = Q6_W_vdeal_VVR(*(const HVX_Vector *)(r1 + X + 128), *(const HVX_Vector *)(r1 + X), -1);
            HVX_VectorPair e0 = Q6_Wh_vsxt_Vb(Q6_V_lo_W(d0));
            HVX_VectorPair o0 = Q6_Wh_vsxt_Vb(Q6_V_hi_W(d0));
            HVX_VectorPair e1 = Q6_Wh_vsxt_Vb(Q6_V_lo_W(d1));
            HVX_VectorPair o1 = Q6_Wh_vsxt_Vb(Q6_V_hi_W(d1));
            HVX_Vector slo = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(Q6_V_lo_W(e0), Q6_V_lo_W(o0)),
                                             Q6_Vh_vadd_VhVh(Q6_V_lo_W(e1), Q6_V_lo_W(o1)));
            HVX_Vector shi = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(Q6_V_hi_W(e0), Q6_V_hi_W(o0)),
                                             Q6_Vh_vadd_VhVh(Q6_V_hi_W(e1), Q6_V_hi_W(o1)));
            HVX_Vector ql = Q6_Vuh_vlsr_VuhR(Q6_Vh_vabs_Vh(slo), 2);
            HVX_Vector qh = Q6_Vuh_vlsr_VuhR(Q6_Vh_vabs_Vh(shi), 2);
            HVX_Vector poollo = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(vzero, slo), Q6_Vh_vsub_VhVh(vzero, ql), ql);
            HVX_Vector poolhi = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(vzero, shi), Q6_Vh_vsub_VhVh(vzero, qh), qh);
            HVX_Vector rlo = requant_h(poollo, vmult, vhalf, shift, vzp, vzero);
            HVX_Vector rhi = requant_h(poolhi, vmult, vhalf, shift, vzp, vzero);
            *(HVX_Vector *)(o + X/2) = Q6_Vb_vasr_VhVhR_sat(rhi, rlo, 0);
        }
        for (int ox = X/2; ox < ow; ox++) {
            int a=r0[2*ox], b=r0[2*ox+1], c=r1[2*ox], d=r1[2*ox+1];
            o[ox] = ref_element((a+b+c+d)/4, mult, shift, zp);
        }
    }
}

void candidate_kernel(const int8_t *in, int8_t *out, int w, int h,
                      int32_t mult, int shift, int8_t zp) {
    int ow = w / 2, oh = h / 2;
    const uint32_t CH_IN  = (uint32_t)(2 * BR * w);
    const uint32_t CH_OUT = (uint32_t)(BR * ow);
    const uint32_t vt_i0 = VTCM_BASE,             vt_i1 = VTCM_BASE + CH_IN;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH_IN,   vt_o1 = VTCM_BASE + 2*CH_IN + CH_OUT;

    int ntiles = oh / BR;
    int done_rows = ntiles * BR;

    if (ntiles == 0) { hvx_avgpool_requant(in, out, w, h, mult, shift, zp); return; }

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

        hvx_avgpool_requant((const int8_t *)(uintptr_t)cur_i, (int8_t *)(uintptr_t)cur_o, w, 2 * BR, mult, shift, zp);

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
            out[oy*ow+ox] = ref_element((a+b+c+d)/4, mult, shift, zp);
        }
    }
}
