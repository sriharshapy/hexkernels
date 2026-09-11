/* sol_03: u8_sobel_mag â€” HVX with aligned-buffer + valign approach.
 * Copies rows to aligned padded buffers, uses Q6_V_valign_VVR for shifts.
 * Computes Gx, Gy as i16; takes abs+add to get magnitude in i16;
 * stores i16 to temp, then scalar saturate-to-u8.
 */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static uint8_t g_up[256]  __attribute__((aligned(128)));
static uint8_t g_cc[256]  __attribute__((aligned(128)));
static uint8_t g_dn[256]  __attribute__((aligned(128)));
static int16_t g_mag[256] __attribute__((aligned(128)));

static inline int cl(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h) {
    for (int y = 0; y < h; y++) {
        const uint8_t *rup_r = in + (y > 0   ? y-1 : 0)   * w;
        const uint8_t *rc_r  = in + y * w;
        const uint8_t *rdn_r = in + (y < h-1 ? y+1 : h-1) * w;
        uint8_t *ro = out + y * w;

        g_up[0] = rup_r[0]; __builtin_memcpy(g_up+1, rup_r, w); g_up[w+1] = rup_r[w-1];
        g_cc[0] = rc_r[0];  __builtin_memcpy(g_cc+1, rc_r,  w); g_cc[w+1] = rc_r[w-1];
        g_dn[0] = rdn_r[0]; __builtin_memcpy(g_dn+1, rdn_r, w); g_dn[w+1] = rdn_r[w-1];

        int k = 0;
        for (; k + 128 <= w; k += 128) {
            const HVX_Vector *cc0 = (const HVX_Vector *)(g_cc + k);
            const HVX_Vector *cc1 = (const HVX_Vector *)(g_cc + k + 128);
            const HVX_Vector *up0 = (const HVX_Vector *)(g_up + k);
            const HVX_Vector *up1 = (const HVX_Vector *)(g_up + k + 128);
            const HVX_Vector *dn0 = (const HVX_Vector *)(g_dn + k);
            const HVX_Vector *dn1 = (const HVX_Vector *)(g_dn + k + 128);

            /* Extract 9 neighborhood vectors via valign:
             * ul=g_up[k..k+127], uc=g_up[k+1..], ur=g_up[k+2..]
             * cl=g_cc[k..], cc=g_cc[k+1..], cr=g_cc[k+2..]
             * dl=g_dn[k..], dc=g_dn[k+1..], dr=g_dn[k+2..] */
            HVX_Vector ul = *up0;                              /* g_up[k..k+127] */
            HVX_Vector uc = Q6_V_valign_VVR(*up1, *up0, 1);  /* g_up[k+1..k+128] */
            HVX_Vector ur = Q6_V_valign_VVR(*up1, *up0, 2);  /* g_up[k+2..k+129] */
            HVX_Vector cl_v = *cc0;
            HVX_Vector cr_v = Q6_V_valign_VVR(*cc1, *cc0, 2);
            HVX_Vector dl  = *dn0;
            HVX_Vector dc  = Q6_V_valign_VVR(*dn1, *dn0, 1);
            HVX_Vector dr  = Q6_V_valign_VVR(*dn1, *dn0, 2);

            /* Unpack u8->u16 */
            HVX_VectorPair wul = Q6_Wuh_vunpack_Vub(ul);
            HVX_VectorPair wuc = Q6_Wuh_vunpack_Vub(uc);
            HVX_VectorPair wur = Q6_Wuh_vunpack_Vub(ur);
            HVX_VectorPair wcl = Q6_Wuh_vunpack_Vub(cl_v);
            HVX_VectorPair wcr = Q6_Wuh_vunpack_Vub(cr_v);
            HVX_VectorPair wdl = Q6_Wuh_vunpack_Vub(dl);
            HVX_VectorPair wdc = Q6_Wuh_vunpack_Vub(dc);
            HVX_VectorPair wdr = Q6_Wuh_vunpack_Vub(dr);

            /* Gx lo = -ul + ur - 2*cl + 2*cr - dl + dr */
            HVX_Vector gx_lo = Q6_Vh_vadd_VhVh(
                Q6_Vh_vsub_VhVh(Q6_V_lo_W(wur), Q6_V_lo_W(wul)),
                Q6_Vh_vadd_VhVh(
                    Q6_Vh_vsub_VhVh(Q6_Vh_vasl_VhR(Q6_V_lo_W(wcr),1),
                                    Q6_Vh_vasl_VhR(Q6_V_lo_W(wcl),1)),
                    Q6_Vh_vsub_VhVh(Q6_V_lo_W(wdr), Q6_V_lo_W(wdl))));
            /* Gy lo = -ul - 2*uc - ur + dl + 2*dc + dr */
            HVX_Vector gy_lo = Q6_Vh_vadd_VhVh(
                Q6_Vh_vsub_VhVh(Q6_V_lo_W(wdl), Q6_V_lo_W(wul)),
                Q6_Vh_vadd_VhVh(
                    Q6_Vh_vsub_VhVh(Q6_Vh_vasl_VhR(Q6_V_lo_W(wdc),1),
                                    Q6_Vh_vasl_VhR(Q6_V_lo_W(wuc),1)),
                    Q6_Vh_vsub_VhVh(Q6_V_lo_W(wdr), Q6_V_lo_W(wur))));
            HVX_Vector mag_lo = Q6_Vh_vadd_VhVh(Q6_Vh_vabs_Vh(gx_lo), Q6_Vh_vabs_Vh(gy_lo));

            /* Gx hi */
            HVX_Vector gx_hi = Q6_Vh_vadd_VhVh(
                Q6_Vh_vsub_VhVh(Q6_V_hi_W(wur), Q6_V_hi_W(wul)),
                Q6_Vh_vadd_VhVh(
                    Q6_Vh_vsub_VhVh(Q6_Vh_vasl_VhR(Q6_V_hi_W(wcr),1),
                                    Q6_Vh_vasl_VhR(Q6_V_hi_W(wcl),1)),
                    Q6_Vh_vsub_VhVh(Q6_V_hi_W(wdr), Q6_V_hi_W(wdl))));
            /* Gy hi */
            HVX_Vector gy_hi = Q6_Vh_vadd_VhVh(
                Q6_Vh_vsub_VhVh(Q6_V_hi_W(wdl), Q6_V_hi_W(wul)),
                Q6_Vh_vadd_VhVh(
                    Q6_Vh_vsub_VhVh(Q6_Vh_vasl_VhR(Q6_V_hi_W(wdc),1),
                                    Q6_Vh_vasl_VhR(Q6_V_hi_W(wuc),1)),
                    Q6_Vh_vsub_VhVh(Q6_V_hi_W(wdr), Q6_V_hi_W(wur))));
            HVX_Vector mag_hi = Q6_Vh_vadd_VhVh(Q6_Vh_vabs_Vh(gx_hi), Q6_Vh_vabs_Vh(gy_hi));

            /* Store i16 magnitude to aligned temp */
            *(HVX_Vector *)(g_mag + k)      = mag_lo;
            *(HVX_Vector *)(g_mag + k + 64) = mag_hi;
        }

        /* Convert HVX i16 to u8 with min-255 saturation, store to output */
        for (int kk = 0; kk < k; kk++) {
            int v = (int)g_mag[kk];
            ro[kk] = (uint8_t)(v > 255 ? 255 : v);
        }

        /* Scalar tail */
        for (; k < w; k++) {
            int xl = (k > 0)   ? k-1 : 0;
            int xr = (k < w-1) ? k+1 : w-1;
            int gx = -(int)rup_r[xl] + (int)rup_r[xr]
                     -2*(int)rc_r[xl] + 2*(int)rc_r[xr]
                     -(int)rdn_r[xl]  + (int)rdn_r[xr];
            int gy = -(int)rup_r[xl] - 2*(int)rup_r[k] - (int)rup_r[xr]
                     +(int)rdn_r[xl]  + 2*(int)rdn_r[k] + (int)rdn_r[xr];
            int mag = (gx<0?-gx:gx) + (gy<0?-gy:gy);
            ro[k] = (uint8_t)(mag > 255 ? 255 : mag);
        }
    }
}