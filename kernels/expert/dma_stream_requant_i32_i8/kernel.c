/* EXPERT (=solutions/s1.c) -- hvx + dma + vtcm. DMA double-buffers int32
 * input tiles (CH=16384 bytes = 4096 elements) from DDR into VTCM, requant on
 * -chip with the sign-aware round-half-away trick, and DMAs the int8 result
 * tile (CH/4 = 4096 bytes) straight back out. Double-buffered so tile c+1's
 * input DMA overlaps tile c's requant+output-DMA. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384                  /* bytes per input tile = 4096 int32 elements */

static desc_t d_in, d_pf, d_out;

static inline HVX_Vector requant_vec(HVX_Vector v, HVX_Vector vmult, HVX_Vector vhalf,
                                     HVX_Vector vzp32, int shift) {
    HVX_Vector sm  = Q6_Vw_vasr_VwR(v, 31);
    HVX_Vector abs = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(v, sm), sm);
    HVX_Vector am  = Q6_Vw_vmpyie_VwVuh(abs, vmult);
    HVX_Vector sh  = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(am, vhalf), shift);
    HVX_Vector r   = Q6_Vw_vadd_VwVw(Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(sh, sm), sm), vzp32);
    return r;
}

void candidate_kernel(const int32_t *a, int8_t *out, int n, int32_t mult, int shift, int8_t zp) {
    const uint32_t vt_i0 = VTCM_BASE,           vt_i1 = VTCM_BASE + CH;
    const uint32_t vt_o0 = VTCM_BASE + 2 * CH,  vt_o1 = vt_o0 + CH / 4;

    int elems_per_tile = CH / 4;     /* 4096 */
    int nfull = n / elems_per_tile;
    int rem   = n - nfull * elems_per_tile;

    int32_t half = (shift > 0) ? (1 << (shift - 1)) : 0;
    HVX_Vector vmult = Q6_V_vsplat_R((int32_t)(uint16_t)mult);
    HVX_Vector vhalf = Q6_V_vsplat_R(half);
    HVX_Vector vzp32 = Q6_V_vsplat_R((int32_t)zp);

    if (nfull == 0) {
        for (int i = 0; i < n; i++) {
            int64_t v = a[i];
            int64_t absv = v < 0 ? -v : v;
            int64_t am = absv * (int64_t)mult;
            int64_t sh = (am + half) >> shift;
            int64_t r = (v < 0) ? -sh : sh;
            r += zp;
            if (r > 127) r = 127; if (r < -128) r = -128;
            out[i] = (int8_t)r;
        }
        return;
    }

    d_in.next = 0; d_in.ctrl = CH;
    d_in.src = (uint32_t)(uintptr_t)a; d_in.dst = vt_i0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_i = (c & 1) ? vt_i1 : vt_i0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_i = (c & 1) ? vt_i0 : vt_i1;

        if (c + 1 < nfull) {
            d_pf.next = 0; d_pf.ctrl = CH;
            d_pf.src = (uint32_t)(uintptr_t)(a + (c + 1) * elems_per_tile);
            d_pf.dst = nxt_i;
            Q6_dmstart_A(&d_pf);
        }

        HVX_Vector *pi = (HVX_Vector *)(uintptr_t)cur_i;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        int nvec_in = CH / 128;         /* 128 int32 input vectors per tile */
        for (int ov = 0; ov < nvec_in / 4; ov++) {
            HVX_Vector v0 = requant_vec(pi[ov*4+0], vmult, vhalf, vzp32, shift);
            HVX_Vector v1 = requant_vec(pi[ov*4+1], vmult, vhalf, vzp32, shift);
            HVX_Vector v2 = requant_vec(pi[ov*4+2], vmult, vhalf, vzp32, shift);
            HVX_Vector v3 = requant_vec(pi[ov*4+3], vmult, vhalf, vzp32, shift);
            HVX_Vector h_lo = Q6_Vh_vpack_VwVw_sat(v1, v0);
            HVX_Vector h_hi = Q6_Vh_vpack_VwVw_sat(v3, v2);
            po[ov] = Q6_Vb_vpack_VhVh_sat(h_hi, h_lo);
        }

        if (c + 1 < nfull) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH / 4;
        d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * elems_per_tile);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * elems_per_tile; i < nfull * elems_per_tile + rem; i++) {
        int64_t v = a[i];
        int64_t absv = v < 0 ? -v : v;
        int64_t am = absv * (int64_t)mult;
        int64_t sh = (am + half) >> shift;
        int64_t r = (v < 0) ? -sh : sh;
        r += zp;
        if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
