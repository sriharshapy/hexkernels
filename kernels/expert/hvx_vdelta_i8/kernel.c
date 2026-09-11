#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    HVX_Vector vctrl = Q6_Vb_vsplat_R(0xFF);
    for (int i = 0; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        *(HVX_Vector *)(out + i) = Q6_V_vdelta_VV(va, vctrl);
    }
}
