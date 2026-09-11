/* expert: HVX sign-extending unpack via Q6_Wh_vunpack_Vb.
 * Per 128-byte input chunk: Q6_V_lo_W(pair) = sign-extend of INPUT bytes
 * [0..63] -> output int16 [0..63]; Q6_V_hi_W(pair) = sign-extend of INPUT
 * bytes [64..127] -> output int16 [64..127]. (Empirically pinned: lo/hi
 * split the LOW/HIGH halves of the source vector, not interleaved.) */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int16_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_VectorPair p = Q6_Wh_vunpack_Vb(va);
        *(HVX_Vector *)(out + i)      = Q6_V_lo_W(p);
        *(HVX_Vector *)(out + i + 64) = Q6_V_hi_W(p);
    }
    for (; i < n; i++)
        out[i] = (int16_t)(int8_t)a[i];
}
