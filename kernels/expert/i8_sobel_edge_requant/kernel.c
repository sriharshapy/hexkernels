/*
 * i8_sobel_edge_requant HVX candidate kernel (v15 - use VhVh multiply)
 *
 * W=66, H=66.
 *
 * Fix: use Q6_Vh_vmpyi_VhVh (vector×vector) instead of Q6_Vh_vmpyi_VhRb (vector×scalar_byte).
 * Splat mult to a vector first, then multiply element-wise.
 *
 * This avoids the Q6_Vh_vmpyi_VhRb issue (which produced 0 output).
 */

#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef long HEXAGON_Vect_UN
    __attribute__((__vector_size__(128))) __attribute__((aligned(4)));
#define vmemu(A) (*((HEXAGON_Vect_UN *)(A)))

static inline HVX_Vector sobel_mag64(
    const uint8_t *buf0, const uint8_t *buf1, const uint8_t *buf2, int offset)
{
    HVX_Vector v0l = (HVX_Vector)vmemu(buf0 + offset);
    HVX_Vector v0c = (HVX_Vector)vmemu(buf0 + offset + 1);
    HVX_Vector v0r = (HVX_Vector)vmemu(buf0 + offset + 2);
    HVX_Vector v1l = (HVX_Vector)vmemu(buf1 + offset);
    HVX_Vector v1r = (HVX_Vector)vmemu(buf1 + offset + 2);
    HVX_Vector v2l = (HVX_Vector)vmemu(buf2 + offset);
    HVX_Vector v2c = (HVX_Vector)vmemu(buf2 + offset + 1);
    HVX_Vector v2r = (HVX_Vector)vmemu(buf2 + offset + 2);

    HVX_Vector r0l = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v0l));
    HVX_Vector r0c = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v0c));
    HVX_Vector r0r = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v0r));
    HVX_Vector r1l = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v1l));
    HVX_Vector r1r = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v1r));
    HVX_Vector r2l = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v2l));
    HVX_Vector r2c = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v2c));
    HVX_Vector r2r = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v2r));

    /* Gx = (r0r-r0l) + 2*(r1r-r1l) + (r2r-r2l) */
    HVX_Vector d0 = Q6_Vh_vsub_VhVh(r0r, r0l);
    HVX_Vector d1 = Q6_Vh_vsub_VhVh(r1r, r1l);
    HVX_Vector d2 = Q6_Vh_vsub_VhVh(r2r, r2l);
    HVX_Vector gx = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(d0, d2),
                                      Q6_Vh_vadd_VhVh(d1, d1));

    /* Gy = (r2l-r0l) + 2*(r2c-r0c) + (r2r-r0r) */
    HVX_Vector el = Q6_Vh_vsub_VhVh(r2l, r0l);
    HVX_Vector ec = Q6_Vh_vsub_VhVh(r2c, r0c);
    HVX_Vector er = Q6_Vh_vsub_VhVh(r2r, r0r);
    HVX_Vector gy = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(el, er),
                                      Q6_Vh_vadd_VhVh(ec, ec));

    return Q6_Vh_vadd_VhVh(Q6_Vh_vabs_Vh(gx), Q6_Vh_vabs_Vh(gy));
}

void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h,
                      int mult, int shift, int zp)
{
    int half = (shift > 0) ? (1 << (shift - 1)) : 0;
    HVX_Vector vh     = Q6_Vh_vsplat_R(half);
    HVX_Vector vz     = Q6_Vh_vsplat_R(zp);
    HVX_Vector vmult  = Q6_Vh_vsplat_R(mult);   /* broadcast mult to int16 lanes */

    for (int y = 0; y < h; y++) {
        const uint8_t *row0 = in + ((y > 0)     ? (y - 1) : 0)     * w;
        const uint8_t *row1 = in +  y                               * w;
        const uint8_t *row2 = in + ((y < h - 1) ? (y + 1) : h - 1) * w;

        uint8_t buf0[200] __attribute__((aligned(4)));
        uint8_t buf1[200] __attribute__((aligned(4)));
        uint8_t buf2[200] __attribute__((aligned(4)));

        buf0[0] = row0[0];
        __builtin_memcpy(buf0 + 1, row0, w);
        for (int i = w + 1; i < 200; i++) buf0[i] = row0[w - 1];

        buf1[0] = row1[0];
        __builtin_memcpy(buf1 + 1, row1, w);
        for (int i = w + 1; i < 200; i++) buf1[i] = row1[w - 1];

        buf2[0] = row2[0];
        __builtin_memcpy(buf2 + 1, row2, w);
        for (int i = w + 1; i < 200; i++) buf2[i] = row2[w - 1];

        HVX_Vector mag_lo = sobel_mag64(buf0, buf1, buf2, 0);
        HVX_Vector mag_hi = sobel_mag64(buf0, buf1, buf2, 64);

        /*
         * Requantize: v = (mag * mult + half) >> shift + zp
         * Use Q6_Vh_vmpyi_VhVh (int16×int16->int16, lower 16 bits).
         * Max mag=2040, max mult=2, product=4080 < 32767 (int16 max). Fine.
         */
        HVX_Vector req_lo = Q6_Vh_vadd_VhVh(
            Q6_Vh_vasr_VhR(
                Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(mag_lo, vmult), vh),
                shift),
            vz);
        HVX_Vector req_hi = Q6_Vh_vadd_VhVh(
            Q6_Vh_vasr_VhR(
                Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(mag_hi, vmult), vh),
                shift),
            vz);

        /* Pack int16 -> uint8 with unsigned saturation */
        HVX_Vector result = Q6_Vub_vpack_VhVh_sat(req_hi, req_lo);

        uint8_t tmp[128] __attribute__((aligned(128)));
        *(HVX_Vector *)tmp = result;
        __builtin_memcpy(out + y * w, tmp, w);
    }
}
