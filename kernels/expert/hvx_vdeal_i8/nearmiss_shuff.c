/* NEAR-MISS: uses vshuff (the INVERSE permutation) instead of vdeal.
 * Plausible confusion since they are exact inverses. Compiles fine but
 * produces the interleave instead of the de-interleave. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    for (int i = 0; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        *(HVX_Vector *)(out + i) = Q6_Vb_vshuff_Vb(va);   /* WRONG: inverse op */
    }
}
