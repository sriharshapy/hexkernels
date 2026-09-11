/* NEAR-MISS: truncating shift instead of round-half-away-from-zero. Plain
 * arithmetic right shift (floor toward -inf), no +half rounding, no sign-magnitude
 * -> wrong on rounding ties and negatives. Baked MULT=3, ZP=0, sat8 pack match. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define MULT  3
#define SHIFT 4
#define ZP    0

void candidate_kernel(const int16_t *a, int8_t *out, int n) {
    uint16_t m16 = (uint16_t)(int16_t)MULT;
    int32_t  mult_r = (int32_t)((uint32_t)m16 | ((uint32_t)m16 << 16));
    HVX_Vector vzp = Q6_V_vsplat_R((int32_t)ZP);
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_VectorPair w01 = Q6_Ww_vunpack_Vh(*(const HVX_Vector*)(a+i));
        HVX_VectorPair w23 = Q6_Ww_vunpack_Vh(*(const HVX_Vector*)(a+i+64));
        HVX_Vector wv[4] = { Q6_V_lo_W(w01), Q6_V_hi_W(w01), Q6_V_lo_W(w23), Q6_V_hi_W(w23) };
        HVX_Vector r[4];
        for (int j = 0; j < 4; j++) {
            HVX_Vector vm = Q6_Vw_vmpyi_VwRh(wv[j], mult_r);
            r[j] = Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(vm, SHIFT), vzp); /* truncate */
        }
        HVX_Vector ph01 = Q6_Vh_vpack_VwVw_sat(r[1], r[0]);
        HVX_Vector ph23 = Q6_Vh_vpack_VwVw_sat(r[3], r[2]);
        *(HVX_Vector *)(out + i) = Q6_Vb_vpack_VhVh_sat(ph23, ph01);
    }
    for (; i < n; i++) {
        int32_t v = (int32_t)a[i] * MULT;
        int32_t r = (v >> SHIFT) + ZP;
        if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
