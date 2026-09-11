/* NEAR-MISS: swaps the Vu/Vv argument order (passes b as Vu and a as Vv)
 * -- a plausible mix-up given the {Vv:Vu} ordering is easy to misremember.
 * Compiles fine but reads the window from {Vu:Vv} = {a:b} instead of
 * {b:a}, giving the wrong bytes for any RT where the window straddles
 * the block boundary (which it does here, RT=37). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#define RT 37

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_V_valign_VVR(vb, va, RT);  /* WRONG: swapped */
    }
}
