/* i8_weighted_sum sol_01: HVX dot product using Q6_Vw_vrmpyacc_VwVbVb.
   Identical to i8_dot_i32 â€” weighted sum IS a dot product. */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

static int32_t hsum_Vw(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

void candidate_kernel(const int8_t *a, const int8_t *w, int n, int32_t *out) {
    const int vlen = 128;
    HVX_Vector acc = Q6_V_vzero();
    int i = 0;
    for (; i + vlen <= n; i += vlen) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vw = *(const HVX_Vector *)(w + i);
        acc = Q6_Vw_vrmpyacc_VwVbVb(acc, va, vw);
    }
    int32_t s = hsum_Vw(acc);
    for (; i < n; i++) s += (int32_t)a[i] * (int32_t)w[i];
    *out = s;
}