#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int g) {
    for (int k = 0; k < g; k++) {
        HVX_Vector va = *(const HVX_Vector *)(a + k*64);
        HVX_Vector vb = *(const HVX_Vector *)(b + k*64);
        *(HVX_Vector *)(out + k*128) = Q6_Vb_vpack_VhVh_sat(va, vb);
    }
}
