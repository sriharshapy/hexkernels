/* NEAR-MISS: uses valign (the mirror-image shift direction) instead of
 * vlalign -- a plausible mix-up between the two "sibling" align ops.
 * Compiles fine but shifts the window the wrong way for any RT != 0. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#define RT 19

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_V_valign_VVR(va, vb, RT);  /* WRONG: valign not vlalign */
    }
}
