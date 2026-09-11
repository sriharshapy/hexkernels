/* NEAR-MISS: omits the fused ReLU. Requantize is otherwise exact (round half away,
 * sat8), but negative results are NOT clamped to zp -> wrong on every element whose
 * requantized value falls below zp. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define MULT  13
#define SHIFT 3
#define ZP    0

void candidate_kernel(const int32_t *a, int8_t *out, int n) {
    uint16_t m16 = (uint16_t)(int16_t)MULT;
    int32_t  mult_r = (int32_t)((uint32_t)m16 | ((uint32_t)m16 << 16));
    HVX_Vector vhalf = Q6_V_vsplat_R((SHIFT > 0) ? (1 << (SHIFT - 1)) : 0);
    HVX_Vector vzp   = Q6_V_vsplat_R((int32_t)ZP);
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector r[4];
        for (int j = 0; j < 4; j++) {
            HVX_Vector vm = Q6_Vw_vmpyi_VwRh(*(const HVX_Vector*)(a+i+j*32), mult_r);
            HVX_Vector sg = Q6_Vw_vasr_VwR(vm, 31);
            HVX_Vector ab = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(vm, sg), sg);
            HVX_Vector rr = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(ab, vhalf), SHIFT);
            rr = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(rr, sg), sg);
            r[j] = Q6_Vw_vadd_VwVw(rr, vzp);       /* no ReLU */
        }
        HVX_Vector ph01 = Q6_Vh_vpack_VwVw_sat(r[1], r[0]);
        HVX_Vector ph23 = Q6_Vh_vpack_VwVw_sat(r[3], r[2]);
        *(HVX_Vector *)(out + i) = Q6_Vb_vpack_VhVh_sat(ph23, ph01);
    }
    for (; i < n; i++) {
        int64_t v = (int64_t)a[i] * MULT;
        int64_t half = (SHIFT > 0) ? ((int64_t)1 << (SHIFT - 1)) : 0;
        int64_t r = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
        r += ZP;
        if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
