/* EXPERT (achievability bar) — HVX vectorized conditional select.
 * out[i] = mask[i] ? a[i] : b[i]; mask is nonzero-true (uint8).
 * Build a byte predicate where mask>0 (unsigned) and vmux a vs b. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, const uint8_t *mask,
                      int8_t *out, int n) {
    const int vlen = sizeof(HVX_Vector); /* 128 */
    HVX_Vector vz = Q6_V_vzero();
    int i = 0;
    for (; i + vlen <= n; i += vlen) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        HVX_Vector vm = *(const HVX_Vector *)(mask + i);
        HVX_VectorPred p = Q6_Q_vcmp_gt_VubVub(vm, vz); /* nonzero (unsigned) */
        *(HVX_Vector *)(out + i) = Q6_V_vmux_QVV(p, va, vb);
    }
    for (; i < n; i++) out[i] = mask[i] ? a[i] : b[i];
}
