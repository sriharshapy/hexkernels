/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams int32 a[] from DDR into VTCM via uDMA, computes the int32->uint8
 * requantize on the on-chip copy with HVX, and DMAs the uint8 results back.
 * Double-buffered: the next input tile is prefetched (async DMA) while the current
 * tile computes, hiding DDR latency on this bandwidth-heavy task (4B read/1B write
 * per element). Word-lane requant + order-preserving vpacke word->byte pack. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH_EL  4096
#define CH_IN  (CH_EL * 4)
#define CH_OUT (CH_EL)
static desc_t d_in, d_out, d_pf;

static inline HVX_Vector requant_word(HVX_Vector w, int mult, HVX_Vector vhalf, int shift,
                                      HVX_Vector vzp, HVX_Vector vlo, HVX_Vector vhi,
                                      HVX_Vector vzero) {
    HVX_Vector v    = Q6_Vw_vmpyi_VwRh(w, (mult & 0xFFFF) | ((mult & 0xFFFF) << 16));
    HVX_Vector absv = Q6_Vw_vabs_Vw(v);
    HVX_Vector sh   = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(absv, vhalf), shift);
    HVX_Vector negsh= Q6_Vw_vsub_VwVw(vzero, sh);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VwVw(vzero, v);
    HVX_Vector r    = Q6_V_vmux_QVV(neg, negsh, sh);
    r = Q6_Vw_vadd_VwVw(r, vzp);
    r = Q6_Vw_vmax_VwVw(r, vlo);
    r = Q6_Vw_vmin_VwVw(r, vhi);
    return r;
}
static inline HVX_Vector pack4(HVX_Vector r0, HVX_Vector r1, HVX_Vector r2, HVX_Vector r3) {
    HVX_Vector h0 = Q6_Vh_vpacke_VwVw(r1, r0);
    HVX_Vector h1 = Q6_Vh_vpacke_VwVw(r3, r2);
    return Q6_Vb_vpacke_VhVh(h1, h0);
}
static uint8_t ref_scalar(int32_t ai, int32_t mult, int shift, uint8_t zp) {
    int64_t v = (int64_t)ai * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += (int64_t)(uint64_t)zp; if (r > 255) r = 255; if (r < 0) r = 0; return (uint8_t)r;
}

void candidate_kernel(const int32_t *a, uint8_t *out, int n,
                      int32_t mult, int shift, uint8_t zp) {
    int hv = (shift > 0) ? (1 << (shift - 1)) : 0;
    HVX_Vector vhalf = Q6_V_vsplat_R((uint32_t)hv);
    HVX_Vector vzp = Q6_V_vsplat_R((uint32_t)(int32_t)zp);
    HVX_Vector vlo = Q6_V_vsplat_R((uint32_t)(int32_t)(0));
    HVX_Vector vhi = Q6_V_vsplat_R((uint32_t)(int32_t)(255));
    HVX_Vector vzero = Q6_V_vzero();

    const uint32_t vt_x0 = VTCM_BASE,           vt_x1 = VTCM_BASE + CH_IN;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH_IN, vt_o1 = VTCM_BASE + 2*CH_IN + CH_OUT;

    int nfull = n / CH_EL;
    int rem   = n - nfull * CH_EL;

    if (nfull == 0) { for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i], mult, shift, zp); return; }

    d_in.next = 0; d_in.ctrl = CH_IN; d_in.src = (uint32_t)(uintptr_t)a; d_in.dst = vt_x0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    const int nbytevec = CH_EL / 128;
    for (int c = 0; c < nfull; c++) {
        uint32_t cur_x = (c & 1) ? vt_x1 : vt_x0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_x = (c & 1) ? vt_x0 : vt_x1;

        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH_IN;
            d_pf.src = (uint32_t)(uintptr_t)(a + (c + 1) * CH_EL); d_pf.dst = nxt_x;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *px = (HVX_Vector *)(uintptr_t)cur_x;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int vb = 0; vb < nbytevec; vb++) {
            HVX_Vector r0 = requant_word(px[4*vb+0], mult, vhalf, shift, vzp, vlo, vhi, vzero);
            HVX_Vector r1 = requant_word(px[4*vb+1], mult, vhalf, shift, vzp, vlo, vhi, vzero);
            HVX_Vector r2 = requant_word(px[4*vb+2], mult, vhalf, shift, vzp, vlo, vhi, vzero);
            HVX_Vector r3 = requant_word(px[4*vb+3], mult, vhalf, shift, vzp, vlo, vhi, vzero);
            po[vb] = pack4(r0, r1, r2, r3);
        }

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH_OUT; d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH_EL);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH_EL; i < nfull * CH_EL + rem; i++) out[i] = ref_scalar(a[i], mult, shift, zp);
}
