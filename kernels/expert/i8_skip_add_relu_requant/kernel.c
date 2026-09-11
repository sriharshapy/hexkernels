/*
 * sol_03: i8_skip_add_relu_requant -- HVX Q6_Wh_vunpack_Vb widen, add, relu, requantize.
 *
 * Uses HVX_VectorPair for widening i8â†’i16 then i16â†’i32. After relu, sum in [0,254].
 * mult <= 63 (harness), so sum*mult <= 16002, fits int32. Sign-restore rounding.
 *
 * Q6_Wh_vunpack_Vb(va): sign-extends 128 i8 â†’ 128 i16 in a VectorPair (256 bytes).
 *   lo half = i16 for bytes 0..63, hi half = i16 for bytes 64..127.
 * Q6_Ww_vunpack_Vh(vh): sign-extends 64 i16 â†’ 64 i32 in a VectorPair.
 *   lo = i32 for i16[0..31], hi = i32 for i16[32..63].
 * Include <hvx_hexagon_protos.h> for VectorPair intrinsics.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp)
{
    uint16_t m16    = (uint16_t)(int16_t)mult;
    int32_t  mult_r = (int32_t)((uint32_t)m16 | ((uint32_t)m16 << 16));

    int32_t half_val = (shift > 0) ? (1 << (shift - 1)) : 0;
    HVX_Vector vhalf = Q6_V_vsplat_R(half_val);
    HVX_Vector vzp   = Q6_V_vsplat_R((int32_t)zp);
    HVX_Vector vzero = Q6_V_vzero();

    int i = 0;
    /* Process 128 i8 elements per iteration */
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);

        /* widen i8 â†’ i16 (VectorPair: lo=bytes 0..63, hi=bytes 64..127) */
        HVX_VectorPair wa16 = Q6_Wh_vunpack_Vb(va);
        HVX_VectorPair wb16 = Q6_Wh_vunpack_Vb(vb);

        /* add in i16: lo and hi halves separately */
        HVX_Vector s_lo16 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(wa16), Q6_V_lo_W(wb16));
        HVX_Vector s_hi16 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(wa16), Q6_V_hi_W(wb16));

        /* relu in i16 */
        s_lo16 = Q6_Vh_vmax_VhVh(s_lo16, vzero);
        s_hi16 = Q6_Vh_vmax_VhVh(s_hi16, vzero);

        /* widen i16 â†’ i32 */
        HVX_VectorPair ws_lo32 = Q6_Ww_vunpack_Vh(s_lo16);  /* lo=i16[0..31]â†’i32, hi=i16[32..63]â†’i32 */
        HVX_VectorPair ws_hi32 = Q6_Ww_vunpack_Vh(s_hi16);

        /* 4 i32 vectors: covers all 128 elements */
        HVX_Vector v0 = Q6_V_lo_W(ws_lo32);  /* elements   0..31  */
        HVX_Vector v1 = Q6_V_hi_W(ws_lo32);  /* elements  32..63  */
        HVX_Vector v2 = Q6_V_lo_W(ws_hi32);  /* elements  64..95  */
        HVX_Vector v3 = Q6_V_hi_W(ws_hi32);  /* elements  96..127 */

        /* multiply */
        v0 = Q6_Vw_vmpyi_VwRh(v0, mult_r);
        v1 = Q6_Vw_vmpyi_VwRh(v1, mult_r);
        v2 = Q6_Vw_vmpyi_VwRh(v2, mult_r);
        v3 = Q6_Vw_vmpyi_VwRh(v3, mult_r);

        /* sign-restore rounding */
        HVX_Vector sg0 = Q6_Vw_vasr_VwR(v0, 31);
        HVX_Vector sg1 = Q6_Vw_vasr_VwR(v1, 31);
        HVX_Vector sg2 = Q6_Vw_vasr_VwR(v2, 31);
        HVX_Vector sg3 = Q6_Vw_vasr_VwR(v3, 31);

        HVX_Vector ab0 = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(v0, sg0), sg0);
        HVX_Vector ab1 = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(v1, sg1), sg1);
        HVX_Vector ab2 = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(v2, sg2), sg2);
        HVX_Vector ab3 = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(v3, sg3), sg3);

        HVX_Vector r0 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(ab0, vhalf), shift);
        HVX_Vector r1 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(ab1, vhalf), shift);
        HVX_Vector r2 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(ab2, vhalf), shift);
        HVX_Vector r3 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(ab3, vhalf), shift);

        r0 = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(r0, sg0), sg0);
        r1 = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(r1, sg1), sg1);
        r2 = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(r2, sg2), sg2);
        r3 = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(r3, sg3), sg3);

        /* add zp */
        r0 = Q6_Vw_vadd_VwVw(r0, vzp);
        r1 = Q6_Vw_vadd_VwVw(r1, vzp);
        r2 = Q6_Vw_vadd_VwVw(r2, vzp);
        r3 = Q6_Vw_vadd_VwVw(r3, vzp);

        /* pack i32â†’i16â†’i8 with saturation
         * vpack(r1,r0): h[0..31]=sat16(r0.w), h[32..63]=sat16(r1.w) â†’ elements 0..63 in order
         * vpack(r3,r2): h[0..31]=sat16(r2.w), h[32..63]=sat16(r3.w) â†’ elements 64..127
         * vpack(ph23,ph01): b[0..63]=sat8(ph01.h), b[64..127]=sat8(ph23.h)
         */
        HVX_Vector ph01 = Q6_Vh_vpack_VwVw_sat(r1, r0);
        HVX_Vector ph23 = Q6_Vh_vpack_VwVw_sat(r3, r2);
        HVX_Vector pb   = Q6_Vb_vpack_VhVh_sat(ph23, ph01);

        *(HVX_Vector *)(out + i) = pb;
    }

    /* Scalar tail */
    for (; i < n; i++) {
        int32_t sum = (int32_t)a[i] + (int32_t)b[i];
        if (sum < 0) sum = 0;
        int64_t v = (int64_t)sum * (int64_t)mult;
        int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
        r += zp;
        if (r >  127) r =  127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}