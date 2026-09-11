/* Near-miss: forgets to add zp at the end. Since the harness fixes zp=5
 * (nonzero on purpose), every single output element is off by exactly -5 --
 * a robust, total-coverage discriminator. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector requant_vec_nozp(HVX_Vector v, HVX_Vector vmult, HVX_Vector vhalf, int shift) {
    HVX_Vector sm  = Q6_Vw_vasr_VwR(v, 31);
    HVX_Vector abs = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(v, sm), sm);
    HVX_Vector am  = Q6_Vw_vmpyie_VwVuh(abs, vmult);
    HVX_Vector sh  = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(am, vhalf), shift);
    HVX_Vector r   = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(sh, sm), sm);   /* BUG: no + zp */
    return r;
}

void candidate_kernel(const int32_t *a, int8_t *out, int n, int32_t mult, int shift, int8_t zp) {
    (void)zp;
    int32_t half = (shift > 0) ? (1 << (shift - 1)) : 0;
    HVX_Vector vmult = Q6_V_vsplat_R((int32_t)(uint16_t)mult);
    HVX_Vector vhalf = Q6_V_vsplat_R(half);

    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector v0 = requant_vec_nozp(*(const HVX_Vector *)(a + i +  0), vmult, vhalf, shift);
        HVX_Vector v1 = requant_vec_nozp(*(const HVX_Vector *)(a + i + 32), vmult, vhalf, shift);
        HVX_Vector v2 = requant_vec_nozp(*(const HVX_Vector *)(a + i + 64), vmult, vhalf, shift);
        HVX_Vector v3 = requant_vec_nozp(*(const HVX_Vector *)(a + i + 96), vmult, vhalf, shift);
        HVX_Vector h_lo = Q6_Vh_vpack_VwVw_sat(v1, v0);
        HVX_Vector h_hi = Q6_Vh_vpack_VwVw_sat(v3, v2);
        *(HVX_Vector *)(out + i) = Q6_Vb_vpack_VhVh_sat(h_hi, h_lo);
    }
    for (; i < n; i++) {
        int64_t v = a[i];
        int64_t absv = v < 0 ? -v : v;
        int64_t am = absv * (int64_t)mult;
        int64_t sh = (am + half) >> shift;
        int64_t r = (v < 0) ? -sh : sh;
        /* BUG: zp never added */
        if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
