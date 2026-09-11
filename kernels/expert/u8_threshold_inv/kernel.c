/* sol_03: HVX inverse threshold unrolled 2x */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *in, uint8_t *out, int n, uint8_t thresh) {
    const int vlen = 128;
    HVX_Vector vt    = Q6_Vb_vsplat_R((int)thresh);
    HVX_Vector v255  = Q6_Vb_vsplat_R(0xFF);
    HVX_Vector vzero = Q6_V_vzero();
    int i = 0;
    for (; i + 2 * vlen <= n; i += 2 * vlen) {
        HVX_Vector vi0 = *(const HVX_Vector *)(in + i);
        HVX_Vector vi1 = *(const HVX_Vector *)(in + i + vlen);
        *(HVX_Vector *)(out + i)        = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VubVub(vt, vi0), v255, vzero);
        *(HVX_Vector *)(out + i + vlen) = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VubVub(vt, vi1), v255, vzero);
    }
    for (; i + vlen <= n; i += vlen) {
        HVX_Vector vi = *(const HVX_Vector *)(in + i);
        *(HVX_Vector *)(out + i) = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VubVub(vt, vi), v255, vzero);
    }
    for (; i < n; i++)
        out[i] = (in[i] < thresh) ? (uint8_t)255 : (uint8_t)0;
}