/* Near-miss: drops the "+1" round-to-nearest term before the shift (plain
 * truncating halve instead of round-then-halve). Differs from the reference
 * on every odd (a[i]*3) value -- a large fraction of elements. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    const int vlen = 128;
    HVX_Vector three = Q6_Vh_vsplat_R(3);
    int i = 0;
    for (; i + vlen <= n; i += vlen) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_VectorPair w = Q6_Wh_vunpack_Vb(va);
        HVX_Vector lo = Q6_V_lo_W(w), hi = Q6_V_hi_W(w);
        lo = Q6_Vh_vmpyi_VhVh(lo, three);
        hi = Q6_Vh_vmpyi_VhVh(hi, three);
        lo = Q6_Vh_vasr_VhR(lo, 1);   /* missing the +1 round term */
        hi = Q6_Vh_vasr_VhR(hi, 1);
        *(HVX_Vector *)(out + i) = Q6_Vb_vpack_VhVh_sat(hi, lo);
    }
    for (; i < n; i++) {
        int32_t t = (int32_t)a[i] * 3;
        t = t >> 1;
        if (t > 127) t = 127;
        if (t < -128) t = -128;
        out[i] = (int8_t)t;
    }
}
