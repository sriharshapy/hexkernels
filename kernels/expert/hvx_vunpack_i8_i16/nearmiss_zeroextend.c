/* NEAR-MISS: uses the UNSIGNED (zero-extending) unpack Q6_Wuh_vunpack_Vub
 * instead of the signed (sign-extending) Q6_Wh_vunpack_Vb. Compiles and
 * passes for non-negative bytes, but fails bit-exact on negative inputs
 * (e.g. a[i]=-128 should give out[i]=-128, not 128). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int16_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_VectorPair p = Q6_Wuh_vunpack_Vub(va);   /* WRONG: zero-extend */
        *(HVX_Vector *)(out + i)      = Q6_V_lo_W(p);
        *(HVX_Vector *)(out + i + 64) = Q6_V_hi_W(p);
    }
    for (; i < n; i++)
        out[i] = (int16_t)(uint8_t)a[i];   /* WRONG: zero-extend */
}
