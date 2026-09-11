/* sol_04: i8_laplacian3x3 â€” HVX with proper unaligned access via vmemu macro.
 * Uses HEXAGON_Vect_UN type (aligned(4) only) for unaligned loads and stores.
 * For each row, loads u8 rows (possibly unaligned) and processes them with HVX.
 * Unpack u8->u16, compute Laplacian in i16, store i16 results unaligned.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

/* Unaligned HVX vector type - aligned to 4 bytes only */
typedef long HEXAGON_Vect_UN __attribute__((__vector_size__(128))) __attribute__((aligned(4)));
#define vmemu(A) (*((HEXAGON_Vect_UN *)(A)))

static inline int cl(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void candidate_kernel(const uint8_t *in, int16_t *out, int w, int h) {
    for (int y = 0; y < h; y++) {
        int y_up = (y > 0) ? y - 1 : 0;
        int y_dn = (y < h-1) ? y + 1 : h-1;
        const uint8_t *row_c  = in + y    * w;
        const uint8_t *row_up = in + y_up * w;
        const uint8_t *row_dn = in + y_dn * w;
        int16_t *row_out = out + y * w;

        /* Scalar: column 0 */
        row_out[0] = (int16_t)((int)row_up[0] + (int)row_dn[0]
                              + (int)row_c[0]
                              + (int)(w > 1 ? row_c[1] : row_c[0])
                              - 4*(int)row_c[0]);

        /* HVX interior: x in [1, w-1), step 128 */
        int x = 1;
        for (; x + 128 <= w - 1; x += 128) {
            /* Load u8 rows at positions x-1, x, x+1 (128 bytes each) */
            HVX_Vector curr_c  = (HVX_Vector)vmemu(row_c  + x);
            HVX_Vector up_v    = (HVX_Vector)vmemu(row_up + x);
            HVX_Vector dn_v    = (HVX_Vector)vmemu(row_dn + x);
            HVX_Vector left_v  = (HVX_Vector)vmemu(row_c  + x - 1);
            HVX_Vector right_v = (HVX_Vector)vmemu(row_c  + x + 1);

            /* Unpack u8->u16 zero-extend: 128 u8 -> 2x64 u16 */
            HVX_VectorPair wp_curr  = Q6_Wuh_vunpack_Vub(curr_c);
            HVX_VectorPair wp_up    = Q6_Wuh_vunpack_Vub(up_v);
            HVX_VectorPair wp_dn    = Q6_Wuh_vunpack_Vub(dn_v);
            HVX_VectorPair wp_left  = Q6_Wuh_vunpack_Vub(left_v);
            HVX_VectorPair wp_rght  = Q6_Wuh_vunpack_Vub(right_v);

            /* Lo half: pixels x..x+63, result in i16 */
            HVX_Vector sum_lo = Q6_Vh_vadd_VhVh(Q6_V_lo_W(wp_up), Q6_V_lo_W(wp_dn));
            sum_lo = Q6_Vh_vadd_VhVh(sum_lo, Q6_V_lo_W(wp_left));
            sum_lo = Q6_Vh_vadd_VhVh(sum_lo, Q6_V_lo_W(wp_rght));
            sum_lo = Q6_Vh_vsub_VhVh(sum_lo, Q6_Vh_vasl_VhR(Q6_V_lo_W(wp_curr), 2));

            /* Hi half: pixels x+64..x+127 */
            HVX_Vector sum_hi = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wp_up), Q6_V_hi_W(wp_dn));
            sum_hi = Q6_Vh_vadd_VhVh(sum_hi, Q6_V_hi_W(wp_left));
            sum_hi = Q6_Vh_vadd_VhVh(sum_hi, Q6_V_hi_W(wp_rght));
            sum_hi = Q6_Vh_vsub_VhVh(sum_hi, Q6_Vh_vasl_VhR(Q6_V_hi_W(wp_curr), 2));

            /* Unaligned stores of 64 i16 values each */
            vmemu(row_out + x)      = (HEXAGON_Vect_UN)sum_lo;
            vmemu(row_out + x + 64) = (HEXAGON_Vect_UN)sum_hi;
        }

        /* Scalar tail for remaining interior pixels */
        for (; x < w - 1; x++) {
            row_out[x] = (int16_t)((int)row_up[x] + (int)row_dn[x]
                                  + (int)row_c[x-1] + (int)row_c[x+1]
                                  - 4*(int)row_c[x]);
        }

        /* Scalar: column w-1 */
        if (w > 1) {
            int xx = w - 1;
            row_out[xx] = (int16_t)((int)row_up[xx] + (int)row_dn[xx]
                                   + (int)row_c[xx-1] + (int)row_c[xx]
                                   - 4*(int)row_c[xx]);
        }
    }
}