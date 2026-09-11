/* Expert: HVX-vectorized requant (same as solutions/s1.c). */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int32_t *acc, int8_t *out, int n) {
    static int32_t tmp[32] HVX_ALIGN;
    const HVX_Vector eight = Q6_V_vsplat_R(8);
    for (int b = 0; b < n; b += 32) {
        HVX_Vector v   = *(const HVX_Vector *)(acc + b);
        HVX_Vector v17 = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(v, 4), v);
        v17            = Q6_Vw_vadd_VwVw(v17, eight);
        HVX_Vector vr  = Q6_Vw_vasr_VwR(v17, 4);
        *(HVX_Vector *)tmp = vr;
        for (int t = 0; t < 32; t++) {
            int r = tmp[t];
            if (r > 127) r = 127; else if (r < -128) r = -128;
            out[b + t] = (int8_t)r;
        }
    }
}
