#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, const int8_t *c, int8_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        HVX_Vector vc = *(const HVX_Vector *)(c + i);
        HVX_VectorPred qt = Q6_Q_vcmp_gt_VbVb(va, vb);
        *(HVX_Vector *)(out + i) = Q6_V_vmux_QVV(qt, va, vc);
    }
    for (; i < n; i++)
        out[i] = (a[i] > b[i]) ? a[i] : c[i];
}
