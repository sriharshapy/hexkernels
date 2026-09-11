/* NEAR-MISS: truncating shift instead of round-half-away-from-zero.
 * Uses a plain arithmetic right shift (floor toward -inf) with no +half rounding
 * and no sign-magnitude handling -> wrong on rounding ties and on negatives.
 * Everything else (baked MULT=5, ZP=0, sat8 pack) matches. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define MULT  5
#define SHIFT 3
#define ZP    0

void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n) {
    uint16_t m16 = (uint16_t)(int16_t)MULT;
    int32_t  mult_r = (int32_t)((uint32_t)m16 | ((uint32_t)m16 << 16));
    HVX_Vector vzp = Q6_V_vsplat_R((int32_t)ZP);
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector r[4];
        for (int j = 0; j < 4; j++) {
            HVX_Vector s = Q6_Vw_vadd_VwVw(*(const HVX_Vector*)(a+i+j*32),
                                           *(const HVX_Vector*)(b+i+j*32));
            HVX_Vector vm = Q6_Vw_vmpyi_VwRh(s, mult_r);
            r[j] = Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(vm, SHIFT), vzp); /* truncate, no round */
        }
        HVX_Vector ph01 = Q6_Vh_vpack_VwVw_sat(r[1], r[0]);
        HVX_Vector ph23 = Q6_Vh_vpack_VwVw_sat(r[3], r[2]);
        *(HVX_Vector *)(out + i) = Q6_Vb_vpack_VhVh_sat(ph23, ph01);
    }
    for (; i < n; i++) {
        int64_t v = ((int64_t)a[i] + (int64_t)b[i]) * MULT;
        int64_t r = (v >> SHIFT) + ZP;
        if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
