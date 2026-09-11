/* Near-miss: two's-complement wraparound add instead of saturating add. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *op, int8_t *out, int n) {
    const int vlen = 128;
    HVX_Vector vop = *(const HVX_Vector *)op;
    int i = 0;
    for (; i + vlen <= n; i += vlen) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        *(HVX_Vector *)(out + i) = Q6_Vb_vadd_VbVb(va, vop);
    }
    for (; i < n; i++) out[i] = (int8_t)(a[i] + op[i % 128]);
}
