/* expert: HVX negative average via Q6_Vb_vnavg_VbVb. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vb_vnavg_VbVb(va, vb);
    }
    for (; i < n; i++)
        out[i] = (int8_t)(((int)a[i] - (int)b[i]) >> 1);
}
