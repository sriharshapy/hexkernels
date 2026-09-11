/* EXPERT (achievability bar) = solutions/s1.c: HVX vmax-with-zero ReLU. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *x, int8_t *out, int n) {
    const int vlen = sizeof(HVX_Vector);
    HVX_Vector zero = Q6_V_vzero();
    int i = 0;
    for (; i + vlen <= n; i += vlen)
        *(HVX_Vector *)(out + i) = Q6_Vb_vmax_VbVb(*(const HVX_Vector *)(x + i), zero);
    for (; i < n; i++) out[i] = (x[i] > 0) ? x[i] : 0;
}
