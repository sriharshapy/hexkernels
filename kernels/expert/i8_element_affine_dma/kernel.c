/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams a[] from DDR into VTCM via uDMA, computes int8 affine-requantize on the
 * on-chip copy with HVX, DMAs results back. Double-buffered: next input tile is
 * prefetched (async DMA) while the current tile computes, hiding DDR latency.
 * Round-half-away-from-zero via sign-split in halfword lanes (valid while
 * a*scale+shift fits int16). */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384

static desc_t d_in, d_out, d_pf;

static inline HVX_Vector affine_vec(HVX_Vector va, HVX_Vector vscale, HVX_Vector vshift,
                                    HVX_Vector vhalf, int s, HVX_Vector vzero) {
    HVX_VectorPair ah = Q6_Wh_vsxt_Vb(va);
    HVX_Vector res[2];
    HVX_Vector parts[2]; parts[0] = Q6_V_lo_W(ah); parts[1] = Q6_V_hi_W(ah);
    for (int k = 0; k < 2; k++) {
        HVX_Vector v = Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(parts[k], vscale), vshift);
        HVX_Vector absv = Q6_Vh_vabs_Vh(v);
        HVX_Vector sh = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absv, vhalf), s);
        HVX_Vector negsh = Q6_Vh_vsub_VhVh(vzero, sh);
        HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);
        res[k] = Q6_V_vmux_QVV(neg, negsh, sh);
    }
    return Q6_Vb_vasr_VhVhR_sat(res[1], res[0], 0);
}

static int8_t ref_scalar(int8_t ai, int32_t scale, int32_t shift, int s) {
    int64_t v = (int64_t)ai * (int64_t)scale + (int64_t)shift;
    int64_t half = (s > 0) ? ((int64_t)1 << (s - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> s) : -(((-v) + half) >> s);
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

void candidate_kernel(const int8_t *a, int8_t *out, int n,
                      int32_t scale, int32_t shift, int s) {
    uint32_t sw = ((uint32_t)(scale & 0xFFFF) << 16) | (scale & 0xFFFF);
    uint32_t hw = ((uint32_t)(shift & 0xFFFF) << 16) | (shift & 0xFFFF);
    int hv = (s > 0) ? (1 << (s - 1)) : 0;
    uint32_t hlfw = ((uint32_t)(hv & 0xFFFF) << 16) | (hv & 0xFFFF);
    HVX_Vector vscale = Q6_V_vsplat_R(sw);
    HVX_Vector vshift = Q6_V_vsplat_R(hw);
    HVX_Vector vhalf  = Q6_V_vsplat_R(hlfw);
    HVX_Vector vzero  = Q6_V_vzero();

    const uint32_t vt_x0 = VTCM_BASE,        vt_x1 = VTCM_BASE + CH;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH, vt_o1 = VTCM_BASE + 3*CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i], scale, shift, s);
        return;
    }

    d_in.next = 0; d_in.ctrl = CH; d_in.src = (uint32_t)(uintptr_t)a; d_in.dst = vt_x0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_x = (c & 1) ? vt_x1 : vt_x0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_x = (c & 1) ? vt_x0 : vt_x1;

        if (c + 1 < nfull) {
            const int8_t *nx = a + (c + 1) * CH;
            d_pf.next = 0; d_pf.ctrl = CH; d_pf.src = (uint32_t)(uintptr_t)nx; d_pf.dst = nxt_x;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *px = (HVX_Vector *)(uintptr_t)cur_x;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int v = 0; v < CH / 128; v++) po[v] = affine_vec(px[v], vscale, vshift, vhalf, s, vzero);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH; d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++)
        out[i] = ref_scalar(a[i], scale, shift, s);
}
