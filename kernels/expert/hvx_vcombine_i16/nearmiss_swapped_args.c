/* NEAR-MISS: calls vcombine(a, b) (a first, b second) instead of
 * vcombine(b, a) -- the natural-looking but WRONG argument order given
 * vcombine's lo=second-arg/hi=first-arg convention. This swaps a and b
 * in the output entirely. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int g) {
    for (int k = 0; k < g; k++) {
        HVX_Vector va = *(const HVX_Vector *)(a + k*64);
        HVX_Vector vb = *(const HVX_Vector *)(b + k*64);
        HVX_VectorPair p = Q6_W_vcombine_VV(va, vb);   /* WRONG: swapped order */
        *(HVX_Vector *)(out + k*128)      = Q6_V_lo_W(p);
        *(HVX_Vector *)(out + k*128 + 64) = Q6_V_hi_W(p);
    }
}
