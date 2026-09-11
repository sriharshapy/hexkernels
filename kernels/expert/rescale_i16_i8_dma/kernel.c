/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams int16 a[] from DDR into VTCM via uDMA, rescales to int8 on the on-chip
 * copy with HVX, DMAs the int8 results back. Double-buffered: next input tile
 * prefetched (async DMA) while the current tile computes. Params baked
 * (MULT=3, SHIFT=4, ZP=0); round half away from zero; sat8 on the pack. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define MULT  3
#define SHIFT 4
#define ZP    0
#define T     4096
#define TB_IN (T * 2)        /* int16 tile bytes */
#define TB_OUT (T)           /* int8  tile bytes */

static desc_t d_in, d_pf, d_out;

static inline HVX_Vector requant_word(HVX_Vector x, int32_t mult_r,
                                      HVX_Vector vhalf, int shift, HVX_Vector vzp) {
    HVX_Vector vm = Q6_Vw_vmpyi_VwRh(x, mult_r);
    HVX_Vector sg = Q6_Vw_vasr_VwR(vm, 31);
    HVX_Vector ab = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(vm, sg), sg);
    HVX_Vector r  = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(ab, vhalf), shift);
    r = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(r, sg), sg);
    return Q6_Vw_vadd_VwVw(r, vzp);
}

static inline HVX_Vector rescale_block(HVX_Vector h0, HVX_Vector h1, int32_t mult_r,
                                       HVX_Vector vhalf, HVX_Vector vzp) {
    HVX_VectorPair w01 = Q6_Ww_vunpack_Vh(h0);
    HVX_VectorPair w23 = Q6_Ww_vunpack_Vh(h1);
    HVX_Vector r0 = requant_word(Q6_V_lo_W(w01), mult_r, vhalf, SHIFT, vzp);
    HVX_Vector r1 = requant_word(Q6_V_hi_W(w01), mult_r, vhalf, SHIFT, vzp);
    HVX_Vector r2 = requant_word(Q6_V_lo_W(w23), mult_r, vhalf, SHIFT, vzp);
    HVX_Vector r3 = requant_word(Q6_V_hi_W(w23), mult_r, vhalf, SHIFT, vzp);
    HVX_Vector ph01 = Q6_Vh_vpack_VwVw_sat(r1, r0);
    HVX_Vector ph23 = Q6_Vh_vpack_VwVw_sat(r3, r2);
    return Q6_Vb_vpack_VhVh_sat(ph23, ph01);
}

static int8_t ref_scalar(int16_t x) {
    int32_t v = (int32_t)x * MULT;
    int32_t half = (SHIFT > 0) ? (1 << (SHIFT - 1)) : 0;
    int32_t r = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
    r += (int32_t)ZP;
    if (r > 127)  r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

static inline void compute_tile(uint32_t vin, uint32_t vo, int32_t mult_r,
                                HVX_Vector vhalf, HVX_Vector vzp) {
    HVX_Vector *pa = (HVX_Vector *)(uintptr_t)vin;   /* 64 int16 per vector */
    HVX_Vector *po = (HVX_Vector *)(uintptr_t)vo;
    for (int g = 0; g < T / 128; g++)
        po[g] = rescale_block(pa[g*2], pa[g*2+1], mult_r, vhalf, vzp);
}

void candidate_kernel(const int16_t *a, int8_t *out, int n) {
    uint16_t m16 = (uint16_t)(int16_t)MULT;
    int32_t  mult_r = (int32_t)((uint32_t)m16 | ((uint32_t)m16 << 16));
    HVX_Vector vhalf = Q6_V_vsplat_R((SHIFT > 0) ? (1 << (SHIFT - 1)) : 0);
    HVX_Vector vzp   = Q6_V_vsplat_R((int32_t)ZP);

    const uint32_t vi0 = VTCM_BASE,          vi1 = VTCM_BASE + TB_IN;
    const uint32_t vo0 = VTCM_BASE + 2*TB_IN, vo1 = VTCM_BASE + 2*TB_IN + TB_OUT;

    int nfull = n / T;
    int rem   = n - nfull * T;

    if (nfull == 0) { for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i]); return; }

    d_in.next = 0; d_in.ctrl = TB_IN; d_in.src = (uint32_t)(uintptr_t)a; d_in.dst = vi0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_i = (c & 1) ? vi1 : vi0;
        uint32_t cur_o = (c & 1) ? vo1 : vo0;
        uint32_t nxt_i = (c & 1) ? vi0 : vi1;

        if (c + 1 < nfull) {
            const int16_t *ni = a + (c + 1) * T;
            d_pf.next = 0; d_pf.ctrl = TB_IN; d_pf.src = (uint32_t)(uintptr_t)ni; d_pf.dst = nxt_i;
            Q6_dmstart_A(&d_pf);
        }

        compute_tile(cur_i, cur_o, mult_r, vhalf, vzp);

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = TB_OUT; d_out.src = cur_o;
        d_out.dst = (uint32_t)(uintptr_t)(out + c * T);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * T; i < nfull * T + rem; i++) out[i] = ref_scalar(a[i]);
}
