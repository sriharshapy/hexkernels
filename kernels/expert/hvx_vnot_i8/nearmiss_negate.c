/* NEAR-MISS: implements arithmetic negation (-a, two's complement) instead
 * of bitwise NOT (~a, ones complement). Plausible confusion since -a =
 * (~a)+1. Compiles fine but fails bit-exact on every element (the two
 * ops never agree for int8: ~a is always -a-1). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    int i = 0;
    HVX_Vector vzero = Q6_Vb_vsplat_R(0);
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        *(HVX_Vector *)(out + i) = Q6_Vb_vsub_VbVb(vzero, va);
    }
    for (; i < n; i++)
        out[i] = (int8_t)(-a[i]);
}
