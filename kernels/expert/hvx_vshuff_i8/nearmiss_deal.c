/* NEAR-MISS: uses vdeal (the INVERSE permutation) instead of vshuff.
 * Plausible confusion since they are named/documented side-by-side and
 * are exact inverses of each other. Compiles fine but produces the
 * de-interleave instead of the interleave -- wrong for any non-trivial
 * block. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    for (int i = 0; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        *(HVX_Vector *)(out + i) = Q6_Vb_vdeal_Vb(va);   /* WRONG: inverse op */
    }
}
