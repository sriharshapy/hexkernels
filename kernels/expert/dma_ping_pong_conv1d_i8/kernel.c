/* EXPERT (=solutions/s1.c) -- hvx + dma + vtcm. DMA double-buffers HALO'D
 * int8 tiles (CH=16384 bytes core + 1-byte halo on each side = CH+2 bytes)
 * from DDR into VTCM: while tile c is being convolved on-chip, tile c+1's
 * halo'd region is DMA'd in, hiding DDR latency behind compute. The first
 * tile has no a[-1] (DMA CH+1 bytes, replicate buf[0]=buf[1] after); a tile
 * that butts exactly against n has no a[tile_base+CH] (DMA CH+1 bytes,
 * replicate buf[CH+1]=buf[CH] after) -- handled generally even though this
 * benchmark's chosen N never triggers the latter (remainder != 0). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH 16384                 /* bytes/elements per tile (int8) */

typedef long HEXAGON_Vect_UN
    __attribute__((__vector_size__(128))) __attribute__((aligned(1)));
#define vmemu(A) (*((const HEXAGON_Vect_UN *)(A)))

static desc_t d_in, d_pf, d_out;

static inline HVX_Vector requant_noscale(HVX_Vector v, HVX_Vector vhalf, int shift) {
    HVX_Vector sm  = Q6_Vw_vasr_VwR(v, 31);
    HVX_Vector abs = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(v, sm), sm);
    HVX_Vector sh  = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(abs, vhalf), shift);
    HVX_Vector r   = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(sh, sm), sm);
    return r;
}

/* Halo'd input buffer is CH+2 bytes but every VTCM slot base must stay
 * 128B-aligned (the compute loop stores output vectors with a direct
 * *(HVX_Vector*)ptr = ... which silently truncates to the containing 128B
 * line on an unaligned address -- unlike loads, which vmemu handles safely
 * at any alignment). Pad the input slot stride up to the next 128B
 * multiple so vt_i1/vt_o0/vt_o1 all land 128B-aligned. */
#define INBUF_SZ (((CH + 2 + 127) / 128) * 128)

void candidate_kernel(const int8_t *a, const int8_t *w, int32_t bias, int shift,
                      int8_t *out, int n) {
    const uint32_t vt_i0 = VTCM_BASE,               vt_i1 = VTCM_BASE + INBUF_SZ;
    const uint32_t vt_o0 = VTCM_BASE + 2 * INBUF_SZ, vt_o1 = vt_o0 + CH;

    int nfull = n / CH;
    int rem   = n - nfull * CH;
    int32_t half = (shift > 0) ? (1 << (shift - 1)) : 0;

    HVX_Vector vw0 = Q6_Vh_vsplat_R((int32_t)(int16_t)w[0]);
    HVX_Vector vw1 = Q6_Vh_vsplat_R((int32_t)(int16_t)w[1]);
    HVX_Vector vw2 = Q6_Vh_vsplat_R((int32_t)(int16_t)w[2]);
    HVX_Vector vbias = Q6_V_vsplat_R(bias);
    HVX_Vector vhalf = Q6_V_vsplat_R(half);

#define SCALAR_CONV(i) do { \
        int32_t left   = a[((i) > 0)     ? (i) - 1 : 0]; \
        int32_t center = a[i]; \
        int32_t right  = a[((i) < n - 1) ? (i) + 1 : n - 1]; \
        int32_t sum = (int32_t)w[0]*left + (int32_t)w[1]*center + (int32_t)w[2]*right + bias; \
        int64_t absum = sum < 0 ? -(int64_t)sum : (int64_t)sum; \
        int64_t sh = (absum + half) >> shift; \
        int64_t r = sum < 0 ? -sh : sh; \
        if (r > 127) r = 127; if (r < -128) r = -128; \
        out[i] = (int8_t)r; \
    } while (0)

    if (nfull == 0) {
        for (int i = 0; i < n; i++) SCALAR_CONV(i);
        return;
    }

    /* Prologue: tile 0's halo'd region. tile 0 has no a[-1]: DMA CH+1 bytes
     * (a[0..CH-1] core, one extra byte for the right halo a[CH]) into buf
     * offset 1, then replicate buf[0]=buf[1] for the missing left halo. */
    {
        int no_right0 = (CH >= n);   /* true only if the whole array is 1 tile */
        d_in.next = 0; d_in.ctrl = no_right0 ? CH : (CH + 1);
        d_in.src = (uint32_t)(uintptr_t)a; d_in.dst = vt_i0 + 1;
        Q6_dmstart_A(&d_in); Q6_R_dmwait();
        *(int8_t *)(uintptr_t)vt_i0 = *(int8_t *)(uintptr_t)(vt_i0 + 1);
        if (no_right0)
            *(int8_t *)(uintptr_t)(vt_i0 + CH + 1) = *(int8_t *)(uintptr_t)(vt_i0 + CH);
    }

    int pending_no_right = 0;
    uint32_t pending_nxt_i = 0;

    for (int c = 0; c < nfull; c++) {
        uint32_t cur_i = (c & 1) ? vt_i1 : vt_i0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_i = (c & 1) ? vt_i0 : vt_i1;
        int tile_base = c * CH;

        if (c + 1 < nfull) {
            int nxt_base = (c + 1) * CH;
            /* c+1 >= 1 always here, so it's never tile 0 -> always has a
             * left neighbor a[nxt_base-1]. Right neighbor a[nxt_base+CH]
             * exists iff nxt_base+CH < n. */
            int no_right = (nxt_base + CH >= n);
            d_pf.next = 0; d_pf.ctrl = no_right ? (CH + 1) : (CH + 2);
            d_pf.src = (uint32_t)(uintptr_t)(a + nxt_base - 1);
            d_pf.dst = nxt_i;
            Q6_dmstart_A(&d_pf);
            pending_no_right = no_right; pending_nxt_i = nxt_i;
        }

        int8_t *bufb = (int8_t *)(uintptr_t)cur_i;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;

        for (int v = 0; v < CH / 128; v++) {
            HVX_Vector vl = (HVX_Vector)vmemu(bufb + v*128 + 0);
            HVX_Vector vc = (HVX_Vector)vmemu(bufb + v*128 + 1);
            HVX_Vector vr = (HVX_Vector)vmemu(bufb + v*128 + 2);

            HVX_Vector vl_lo = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vl));
            HVX_Vector vl_hi = Q6_V_hi_W(Q6_Wh_vunpack_Vb(vl));
            HVX_Vector vc_lo = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vc));
            HVX_Vector vc_hi = Q6_V_hi_W(Q6_Wh_vunpack_Vb(vc));
            HVX_Vector vr_lo = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vr));
            HVX_Vector vr_hi = Q6_V_hi_W(Q6_Wh_vunpack_Vb(vr));

            HVX_Vector acc_lo = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(
                Q6_Vh_vmpyi_VhVh(vl_lo, vw0), Q6_Vh_vmpyi_VhVh(vc_lo, vw1)),
                Q6_Vh_vmpyi_VhVh(vr_lo, vw2));
            HVX_Vector acc_hi = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(
                Q6_Vh_vmpyi_VhVh(vl_hi, vw0), Q6_Vh_vmpyi_VhVh(vc_hi, vw1)),
                Q6_Vh_vmpyi_VhVh(vr_hi, vw2));

            HVX_VectorPair w_lo = Q6_Ww_vunpack_Vh(acc_lo);
            HVX_VectorPair w_hi = Q6_Ww_vunpack_Vh(acc_hi);
            HVX_Vector q0 = Q6_V_lo_W(w_lo), q1 = Q6_V_hi_W(w_lo);
            HVX_Vector q2 = Q6_V_lo_W(w_hi), q3 = Q6_V_hi_W(w_hi);

            q0 = Q6_Vw_vadd_VwVw(q0, vbias); q1 = Q6_Vw_vadd_VwVw(q1, vbias);
            q2 = Q6_Vw_vadd_VwVw(q2, vbias); q3 = Q6_Vw_vadd_VwVw(q3, vbias);

            q0 = requant_noscale(q0, vhalf, shift);
            q1 = requant_noscale(q1, vhalf, shift);
            q2 = requant_noscale(q2, vhalf, shift);
            q3 = requant_noscale(q3, vhalf, shift);

            HVX_Vector h_lo = Q6_Vh_vpack_VwVw_sat(q1, q0);
            HVX_Vector h_hi = Q6_Vh_vpack_VwVw_sat(q3, q2);
            po[v] = Q6_Vb_vpack_VhVh_sat(h_hi, h_lo);
        }

        if (c + 1 < nfull) {
            Q6_R_dmwait();
            if (pending_no_right)
                *(int8_t *)(uintptr_t)(pending_nxt_i + CH + 1) =
                    *(int8_t *)(uintptr_t)(pending_nxt_i + CH);
        }

        d_out.next = 0; d_out.ctrl = CH;
        d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + tile_base);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = nfull * CH; i < nfull * CH + rem; i++) SCALAR_CONV(i);
#undef SCALAR_CONV
}
