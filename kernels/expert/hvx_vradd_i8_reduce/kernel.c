#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int32_t *out, int g) {
    HVX_Vector vones = Q6_Vb_vsplat_R(1);
    int nvec = g / 32;
    int i;
    for (i = 0; i < nvec; i++) {
        HVX_Vector va = *(const HVX_Vector *)(a + i * 128);
        *(HVX_Vector *)(out + i * 32) = Q6_Vw_vrmpy_VbVb(va, vones);
    }
    for (int k = nvec * 32; k < g; k++) {
        int32_t sum = 0;
        for (int j = 0; j < 4; j++)
            sum += (int32_t)a[4*k+j];
        out[k] = sum;
    }
}
