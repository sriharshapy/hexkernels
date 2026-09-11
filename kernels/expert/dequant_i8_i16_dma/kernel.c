/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams int8 a[] from DDR into VTCM via uDMA, computes int8->int16 dequantize
 * on the on-chip copy with HVX, and DMAs the int16 results back. Double-buffered:
 * the next input tile is prefetched (async DMA) while the current tile computes,
 * hiding DDR latency on this bandwidth-bound task (1B read/2B write per element).
 * Halfword-lane (x-zp)*scale >> shift; two halfword sub-vectors per input
 * byte-vector via vsxt. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH_EL  8192              /* elements per tile */
#define CH_IN  (CH_EL)           /* input bytes per tile (int8)  */
#define CH_OUT (CH_EL * 2)       /* output bytes per tile (int16) */
static desc_t d_in, d_out, d_pf;

static inline HVX_Vector dequant_half(HVX_Vector xh, HVX_Vector vzp, HVX_Vector vscale, int shift) {
    HVX_Vector d = Q6_Vh_vsub_VhVh(xh, vzp);
    HVX_Vector p = Q6_Vh_vmpyi_VhVh(d, vscale);
    return Q6_Vh_vasr_VhR(p, shift);
}
/* One byte-vector -> pair of natural-order int16 result vectors (vsxt deals, so
 * re-interleave with vshuff). */
static inline HVX_VectorPair dequant_bytevec(HVX_Vector xb, HVX_Vector vzp, HVX_Vector vscale, int shift) {
    HVX_VectorPair xh = Q6_Wh_vsxt_Vb(xb);
    HVX_Vector rlo = dequant_half(Q6_V_lo_W(xh), vzp, vscale, shift);
    HVX_Vector rhi = dequant_half(Q6_V_hi_W(xh), vzp, vscale, shift);
    return Q6_W_vshuff_VVR(rhi, rlo, -2);
}
static int16_t ref_scalar(int8_t x, int8_t zp, int32_t scale, int shift) {
    int32_t v = ((int32_t)x - (int32_t)zp) * scale;
    int32_t r = v >> shift;
    if (r > 32767)  r = 32767;
    if (r < -32768) r = -32768;
    return (int16_t)r;
}

void candidate_kernel(const int8_t *a, int16_t *out, int n,
                      int8_t zp, int32_t scale, int shift) {
    uint32_t sw = ((uint32_t)(scale & 0xFFFF) << 16) | (scale & 0xFFFF);
    uint32_t zw = ((uint32_t)((int32_t)zp & 0xFFFF) << 16) | ((int32_t)zp & 0xFFFF);
    HVX_Vector vscale = Q6_V_vsplat_R(sw);
    HVX_Vector vzp    = Q6_V_vsplat_R(zw);

    const uint32_t vt_x0 = VTCM_BASE,           vt_x1 = VTCM_BASE + CH_IN;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH_IN, vt_o1 = VTCM_BASE + 2*CH_IN + CH_OUT;

    int nfull = n / CH_EL;
    int rem   = n - nfull * CH_EL;

    if (nfull == 0) { for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i], zp, scale, shift); return; }

    d_in.next = 0; d_in.ctrl = CH_IN; d_in.src = (uint32_t)(uintptr_t)a; d_in.dst = vt_x0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    const int nbytevec = CH_EL / 128;    /* input byte-vectors per tile */
    for (int c = 0; c < nfull; c++) {
        uint32_t cur_x = (c & 1) ? vt_x1 : vt_x0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_x = (c & 1) ? vt_x0 : vt_x1;

        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH_IN;
            d_pf.src = (uint32_t)(uintptr_t)(a + (c + 1) * CH_EL); d_pf.dst = nxt_x;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *px = (HVX_Vector *)(uintptr_t)cur_x;          /* byte vectors */
        int16_t   *po16 = (int16_t *)(uintptr_t)cur_o;
        for (int vb = 0; vb < nbytevec; vb++) {
            HVX_VectorPair r = dequant_bytevec(px[vb], vzp, vscale, shift);
            *(HVX_Vector *)(po16 + vb*128)      = Q6_V_lo_W(r);
            *(HVX_Vector *)(po16 + vb*128 + 64) = Q6_V_hi_W(r);
        }

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH_OUT; d_out.src = cur_o;
        d_out.dst = (uint32_t)(uintptr_t)(out + c * CH_EL);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH_EL; i < nfull * CH_EL + rem; i++) out[i] = ref_scalar(a[i], zp, scale, shift);
}
