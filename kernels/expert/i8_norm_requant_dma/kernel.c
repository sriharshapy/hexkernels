/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams int8 a[] from DDR into VTCM via uDMA, computes per-channel normalize +
 * requantize on the on-chip copy with HVX, DMAs the int8 results back.
 * Double-buffered: next input tile prefetched (async DMA) while the current tile
 * computes. Layout [NUM_CH][per_ch]; tile size divides per_ch so each tile lies in a
 * single channel (its NORM_MULT/NORM_SHIFT are constant). Params baked (NUM_CH=16,
 * MULT=5, SHIFT=4, ZP=0); halfword-lane compute; round half away; sat8 on pack. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define NUM_CH 16
#define MULT  5
#define SHIFT 4
#define ZP    0
#define T     4096            /* int8 tile bytes; divides per_ch=32768 (8 tiles/channel) */
#define TB_IN (T)
#define TB_OUT (T)

static const int32_t NORM_MULT[NUM_CH]  = { 3, 5, 7, 9, 11, 13, 15, 17, 3, 5, 7, 9, 11, 13, 15, 17 };
static const int     NORM_SHIFT[NUM_CH] = { 2, 3, 4, 5,  3,  4,  5,  6, 2, 3, 4, 5,  3,  4,  5,  6 };

static desc_t d_in, d_pf, d_out;

static inline HVX_Vector rha_hw(HVX_Vector v, HVX_Vector vhalf, int shift, HVX_Vector vzero) {
    HVX_Vector absv = Q6_Vh_vabs_Vh(v);
    HVX_Vector sh   = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absv, vhalf), shift);
    HVX_Vector negsh = Q6_Vh_vsub_VhVh(vzero, sh);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);
    return Q6_V_vmux_QVV(neg, negsh, sh);
}

static inline HVX_Vector norm_req_hw(HVX_Vector av, HVX_Vector vnm, HVX_Vector vnhalf, int ns,
                                     HVX_Vector vmult, HVX_Vector vhalf, HVX_Vector vzp, HVX_Vector vzero) {
    HVX_Vector nrm = rha_hw(Q6_Vh_vmpyi_VhVh(av, vnm), vnhalf, ns, vzero);
    HVX_Vector r   = rha_hw(Q6_Vh_vmpyi_VhVh(nrm, vmult), vhalf, SHIFT, vzero);
    return Q6_Vh_vadd_VhVh(r, vzp);
}

static int8_t ref_scalar(int8_t ai, int32_t nm, int ns) {
    int64_t nv    = (int64_t)ai * (int64_t)nm;
    int64_t nhalf = ns > 0 ? ((int64_t)1 << (ns - 1)) : 0;
    int64_t norm  = (nv >= 0) ? ((nv + nhalf) >> ns) : -(((-nv) + nhalf) >> ns);
    int64_t v     = norm * (int64_t)MULT;
    int64_t half  = SHIFT > 0 ? ((int64_t)1 << (SHIFT - 1)) : 0;
    int64_t r     = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
    r += ZP;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

static inline void compute_tile(uint32_t vin, uint32_t vo, int32_t nm, int ns,
                                HVX_Vector vmult, HVX_Vector vhalf, HVX_Vector vzp, HVX_Vector vzero) {
    HVX_Vector vnm    = Q6_Vh_vsplat_R(nm);
    HVX_Vector vnhalf = Q6_Vh_vsplat_R((ns > 0) ? (1 << (ns - 1)) : 0);
    HVX_Vector *pa = (HVX_Vector *)(uintptr_t)vin;   /* 128 int8 per vector */
    HVX_Vector *po = (HVX_Vector *)(uintptr_t)vo;
    for (int g = 0; g < T / 128; g++) {
        HVX_VectorPair wa = Q6_Wh_vunpack_Vb(pa[g]);
        HVX_Vector r0 = norm_req_hw(Q6_V_lo_W(wa), vnm, vnhalf, ns, vmult, vhalf, vzp, vzero);
        HVX_Vector r1 = norm_req_hw(Q6_V_hi_W(wa), vnm, vnhalf, ns, vmult, vhalf, vzp, vzero);
        po[g] = Q6_Vb_vpack_VhVh_sat(r1, r0);
    }
}

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    int per_ch = n / NUM_CH;
    HVX_Vector vmult = Q6_Vh_vsplat_R(MULT);
    HVX_Vector vhalf = Q6_Vh_vsplat_R((SHIFT > 0) ? (1 << (SHIFT - 1)) : 0);
    HVX_Vector vzp   = Q6_Vh_vsplat_R(ZP);
    HVX_Vector vzero = Q6_V_vzero();

    const uint32_t vi0 = VTCM_BASE,          vi1 = VTCM_BASE + TB_IN;
    const uint32_t vo0 = VTCM_BASE + 2*TB_IN, vo1 = VTCM_BASE + 2*TB_IN + TB_OUT;

    int nfull = n / T;
    int rem   = n - nfull * T;

    if (nfull == 0) {
        for (int i = 0; i < n; i++) { int c = i / per_ch; out[i] = ref_scalar(a[i], NORM_MULT[c], NORM_SHIFT[c]); }
        return;
    }

    d_in.next = 0; d_in.ctrl = TB_IN; d_in.src = (uint32_t)(uintptr_t)a; d_in.dst = vi0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_i = (c & 1) ? vi1 : vi0;
        uint32_t cur_o = (c & 1) ? vo1 : vo0;
        uint32_t nxt_i = (c & 1) ? vi0 : vi1;

        if (c + 1 < nfull) {
            const int8_t *ni = a + (c + 1) * T;
            d_pf.next = 0; d_pf.ctrl = TB_IN; d_pf.src = (uint32_t)(uintptr_t)ni; d_pf.dst = nxt_i;
            Q6_dmstart_A(&d_pf);
        }

        int ch = (c * T) / per_ch;
        compute_tile(cur_i, cur_o, NORM_MULT[ch], NORM_SHIFT[ch], vmult, vhalf, vzp, vzero);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = TB_OUT; d_out.src = cur_o;
        d_out.dst = (uint32_t)(uintptr_t)(out + c * T);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * T; i < nfull * T + rem; i++) {
        int c = i / per_ch; out[i] = ref_scalar(a[i], NORM_MULT[c], NORM_SHIFT[c]);
    }
}
