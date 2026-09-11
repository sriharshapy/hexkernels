/*
 * requant_per_channel — HVX candidate kernel (optimized)
 *
 * Shape: R=40 rows x C=25 columns. Per-row mult[r] and shift[r].
 *
 * Math:
 *   v = (int64_t)a[r*C+c] * mult[r]
 *   rounded = round_half_away_from_zero(v, shift[r])
 *   out = clamp(rounded + zp, -128, 127)
 *
 * HVX strategy:
 *   - Load C=25 int32s (100B) as one 128B HVX_Vector per row
 *   - Q6_Vw_vmpyi_VwRh: Vd.w[i] = Vu.w[i] * Rt.h[i%2]
 *     Must pack m into BOTH halfwords of scalar: m16 = lo16(m) | (lo16(m)<<16)
 *   - HVX rounding: vabs, vadd, vasr, vmux to restore sign
 *   - Saturate via double vpack_sat
 *   - Output: use union .b[] for byte-granularity extraction, memcpy 25 bytes
 */

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

typedef union {
    HVX_Vector v;
    int8_t     b[128];
} VecB;

void candidate_kernel(const int32_t *a, int8_t *out, int R, int C,
                      const int32_t *mult, const int *shift, int8_t zp) {
    VecB buf __attribute__((aligned(128)));
    HVX_Vector vzero = Q6_V_vsplat_R(0);

    for (int r = 0; r < R; r++) {
        const int32_t *row_a   = a   + r * C;
        int8_t        *row_out = out + r * C;

        int32_t m = mult[r];
        int     s = shift[r];

        /* Load row as 128B vector (unaligned for r > 0) */
        HVX_Vector vdata = *((const HVX_UVector *)row_a);

        /*
         * Multiply: Q6_Vw_vmpyi_VwRh uses Rt.h[i%2] — alternates even/odd halfwords.
         * Pack m into both halfwords so all lanes get the same multiplier.
         */
        int32_t m16 = (int32_t)((uint16_t)(int16_t)m | ((uint32_t)(uint16_t)(int16_t)m << 16));
        HVX_Vector vprod = Q6_Vw_vmpyi_VwRh(vdata, m16);

        /* Round half-away-from-zero */
        HVX_Vector vrounded;
        if (s == 0) {
            vrounded = vprod;
        } else {
            int32_t half = 1 << (s - 1);
            HVX_Vector vhalf  = Q6_V_vsplat_R(half);
            HVX_Vector vabs   = Q6_Vw_vabs_Vw(vprod);
            HVX_Vector vabs_h = Q6_Vw_vadd_VwVw(vabs, vhalf);
            HVX_Vector vshift = Q6_Vw_vasr_VwR(vabs_h, s);

            HVX_VectorPred qneg = Q6_Q_vcmp_gt_VwVw(vzero, vprod);
            HVX_Vector vneg    = Q6_Vw_vsub_VwVw(vzero, vshift);
            vrounded = Q6_V_vmux_QVV(qneg, vneg, vshift);
        }

        /* Add zp, saturate int32->int8 */
        HVX_Vector vzp  = Q6_V_vsplat_R((int32_t)(int8_t)zp);
        HVX_Vector vsum = Q6_Vw_vadd_VwVw(vrounded, vzp);
        HVX_Vector v16  = Q6_Vh_vpack_VwVw_sat(vsum, vsum);
        HVX_Vector v8   = Q6_Vb_vpack_VhVh_sat(v16, v16);

        /* Extract C bytes and store to (unaligned) output row */
        buf.v = v8;
        memcpy(row_out, buf.b, C);
    }
}
