/* expert: HVX unsigned max via Q6_Vub_vmax_VubVub. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vub_vmax_VubVub(va, vb);
    }
    for (; i < n; i++)
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
}
