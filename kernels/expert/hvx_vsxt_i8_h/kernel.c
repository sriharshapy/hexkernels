#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int16_t *out, int n) {
    for (int i = 0; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_VectorPair p = Q6_Wh_vsxt_Vb(va);
        *(HVX_Vector *)(out + i)      = Q6_V_lo_W(p);
        *(HVX_Vector *)(out + i + 64) = Q6_V_hi_W(p);
    }
}
