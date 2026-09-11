/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams int8 in[] from DDR into VTCM via uDMA, computes the GELU-approx (requant
 * index + 256-entry LUT) on the on-chip copy with HVX, and DMAs results back.
 * Double-buffered: the next input tile is prefetched (async DMA) while the current
 * tile computes, hiding DDR latency on this bandwidth-bound task (1B read/1B write
 * per element). Halfword-lane index compute + 128B vlut32 8-pass table (resident
 * in registers). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH 16384
static desc_t d_in, d_out, d_pf;

static void lut_build(const int8_t *L, int8_t *tv0, int8_t *tv1) {
    for (int s = 0; s < 32; s++) {
        tv0[2*s]   = L[s];     tv0[64+2*s] = L[32+s];
        tv0[2*s+1] = L[64+s];  tv0[65+2*s] = L[96+s];
        tv1[2*s]   = L[128+s]; tv1[64+2*s] = L[160+s];
        tv1[2*s+1] = L[192+s]; tv1[65+2*s] = L[224+s];
    }
}
static inline HVX_Vector lut_apply(HVX_Vector vin, HVX_Vector T0, HVX_Vector T1) {
    HVX_Vector a = Q6_Vb_vlut32_VbVbR(vin, T0, 0);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T0, 1);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T0, 2);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T0, 3);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T1, 4);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T1, 5);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T1, 6);
    a = Q6_Vb_vlut32or_VbVbVbR(a, vin, T1, 7);
    return a;
}
static inline HVX_Vector gelu_idx_half(HVX_Vector hv, HVX_Vector vscale,
                                       HVX_Vector v128, HVX_Vector v255, HVX_Vector vzero) {
    HVX_Vector v = Q6_Vh_vmpyi_VhVh(hv, vscale);
    HVX_Vector absv = Q6_Vh_vabs_Vh(v);
    HVX_Vector q = Q6_Vh_vasr_VhR(absv, 6);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);
    HVX_Vector qs = Q6_V_vmux_QVV(neg, Q6_Vh_vsub_VhVh(vzero, q), q);
    HVX_Vector idx = Q6_Vh_vadd_VhVh(qs, v128);
    idx = Q6_Vh_vmax_VhVh(idx, vzero);
    idx = Q6_Vh_vmin_VhVh(idx, v255);
    return idx;
}
static inline HVX_Vector gelu_idx_bytevec(HVX_Vector xb, HVX_Vector vscale,
                                          HVX_Vector v128, HVX_Vector v255, HVX_Vector vzero) {
    HVX_VectorPair xh = Q6_Wh_vsxt_Vb(xb);
    HVX_Vector ilo = gelu_idx_half(Q6_V_lo_W(xh), vscale, v128, v255, vzero);
    HVX_Vector ihi = gelu_idx_half(Q6_V_hi_W(xh), vscale, v128, v255, vzero);
    HVX_VectorPair nat = Q6_W_vshuff_VVR(ihi, ilo, -2);
    return Q6_Vb_vpacke_VhVh(Q6_V_hi_W(nat), Q6_V_lo_W(nat));
}
static int clamp_idx(int x, int scale) {
    int idx = (x * scale) / 64 + 128; if (idx < 0) idx = 0; if (idx > 255) idx = 255; return idx;
}

void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut, int8_t scale) {
    int8_t tv0[128] __attribute__((aligned(128))), tv1[128] __attribute__((aligned(128)));
    lut_build(lut, tv0, tv1);
    HVX_Vector T0 = *(HVX_Vector *)tv0, T1 = *(HVX_Vector *)tv1;
    HVX_Vector vscale = Q6_V_vsplat_R(((uint32_t)((int32_t)scale & 0xFFFF) << 16) | ((int32_t)scale & 0xFFFF));
    HVX_Vector v128 = Q6_V_vsplat_R(0x00800080u), v255 = Q6_V_vsplat_R(0x00FF00FFu), vzero = Q6_V_vzero();

    const uint32_t vt_x0 = VTCM_BASE,        vt_x1 = VTCM_BASE + CH;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH, vt_o1 = VTCM_BASE + 3*CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) { for (int i = 0; i < n; i++) out[i] = lut[clamp_idx((int)in[i], (int)scale)]; return; }

    d_in.next = 0; d_in.ctrl = CH; d_in.src = (uint32_t)(uintptr_t)in; d_in.dst = vt_x0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_x = (c & 1) ? vt_x1 : vt_x0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_x = (c & 1) ? vt_x0 : vt_x1;

        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)(in + (c + 1) * CH); d_pf.dst = nxt_x;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *px = (HVX_Vector *)(uintptr_t)cur_x;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int v = 0; v < CH / 128; v++) {
            HVX_Vector idx = gelu_idx_bytevec(px[v], vscale, v128, v255, vzero);
            po[v] = lut_apply(idx, T0, T1);
        }

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH; d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++) out[i] = lut[clamp_idx((int)in[i], (int)scale)];
}
