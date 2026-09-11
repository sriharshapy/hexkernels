/* sol_03: u8_sobel3x3_x â€” HVX with vmemu unaligned access.
 * Sobel-x = -up_left + up_right - 2*c_left + 2*c_right - dn_left + dn_right
 * Load u8 rows with offset -1 and +1 for left/right neighbours.
 * Unpack to u16, compute in i16 space, store i16 unaligned.
 * Interior blocks: 128 u8 inputs -> 128 i16 outputs per iteration.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef long HEXAGON_Vect_UN __attribute__((__vector_size__(128))) __attribute__((aligned(4)));
#define vmemu(A) (*((HEXAGON_Vect_UN *)(A)))

void candidate_kernel(const uint8_t *in, int16_t *out, int w, int h) {
    for (int y = 0; y < h; y++) {
        const uint8_t *rup = in + (y > 0   ? y-1 : 0)   * w;
        const uint8_t *rc  = in + y * w;
        const uint8_t *rdn = in + (y < h-1 ? y+1 : h-1) * w;
        int16_t *ro = out + y * w;

        /* Scalar left border pixel */
        ro[0] = (int16_t)(-(int)rup[0] + (int)(w>1?rup[1]:rup[0])
                         -2*(int)rc[0]  + 2*(int)(w>1?rc[1]:rc[0])
                         -(int)rdn[0]   + (int)(w>1?rdn[1]:rdn[0]));

        /* HVX interior: x in [1, w-1), 128-pixel blocks */
        int x = 1;
        for (; x + 128 <= w - 1; x += 128) {
            /* Load left/right u8 vectors */
            HVX_Vector up_left  = (HVX_Vector)vmemu(rup + x - 1);
            HVX_Vector up_right = (HVX_Vector)vmemu(rup + x + 1);
            HVX_Vector c_left   = (HVX_Vector)vmemu(rc  + x - 1);
            HVX_Vector c_right  = (HVX_Vector)vmemu(rc  + x + 1);
            HVX_Vector dn_left  = (HVX_Vector)vmemu(rdn + x - 1);
            HVX_Vector dn_right = (HVX_Vector)vmemu(rdn + x + 1);

            /* Unpack u8->u16 */
            HVX_VectorPair wp_ul = Q6_Wuh_vunpack_Vub(up_left);
            HVX_VectorPair wp_ur = Q6_Wuh_vunpack_Vub(up_right);
            HVX_VectorPair wp_cl = Q6_Wuh_vunpack_Vub(c_left);
            HVX_VectorPair wp_cr = Q6_Wuh_vunpack_Vub(c_right);
            HVX_VectorPair wp_dl = Q6_Wuh_vunpack_Vub(dn_left);
            HVX_VectorPair wp_dr = Q6_Wuh_vunpack_Vub(dn_right);

            /* Lo half: Gx = -ul + ur - 2*cl + 2*cr - dl + dr */
            HVX_Vector gx_lo = Q6_Vh_vsub_VhVh(Q6_V_lo_W(wp_ur), Q6_V_lo_W(wp_ul));
            gx_lo = Q6_Vh_vadd_VhVh(gx_lo,
                        Q6_Vh_vsub_VhVh(Q6_Vh_vasl_VhR(Q6_V_lo_W(wp_cr), 1),
                                        Q6_Vh_vasl_VhR(Q6_V_lo_W(wp_cl), 1)));
            gx_lo = Q6_Vh_vadd_VhVh(gx_lo,
                        Q6_Vh_vsub_VhVh(Q6_V_lo_W(wp_dr), Q6_V_lo_W(wp_dl)));

            /* Hi half */
            HVX_Vector gx_hi = Q6_Vh_vsub_VhVh(Q6_V_hi_W(wp_ur), Q6_V_hi_W(wp_ul));
            gx_hi = Q6_Vh_vadd_VhVh(gx_hi,
                        Q6_Vh_vsub_VhVh(Q6_Vh_vasl_VhR(Q6_V_hi_W(wp_cr), 1),
                                        Q6_Vh_vasl_VhR(Q6_V_hi_W(wp_cl), 1)));
            gx_hi = Q6_Vh_vadd_VhVh(gx_hi,
                        Q6_Vh_vsub_VhVh(Q6_V_hi_W(wp_dr), Q6_V_hi_W(wp_dl)));

            vmemu(ro + x)      = (HEXAGON_Vect_UN)gx_lo;
            vmemu(ro + x + 64) = (HEXAGON_Vect_UN)gx_hi;
        }

        /* Scalar tail */
        for (; x < w - 1; x++) {
            ro[x] = (int16_t)(-(int)rup[x-1] + (int)rup[x+1]
                             -2*(int)rc[x-1]  + 2*(int)rc[x+1]
                             -(int)rdn[x-1]   + (int)rdn[x+1]);
        }

        /* Scalar right border */
        if (w > 1) {
            int xr = w-1;
            ro[xr] = (int16_t)(-(int)rup[xr-1] + (int)rup[xr]
                              -2*(int)rc[xr-1]   + 2*(int)rc[xr]
                              -(int)rdn[xr-1]    + (int)rdn[xr]);
        }
    }
}