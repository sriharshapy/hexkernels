/* NEAR-MISS: swaps which half (lo/hi) is stored to the even-slot vs
 * odd-slot of the output -- a plausible mix-up given the naming
 * doesn't make the even/odd mapping obvious. Compiles fine but swaps
 * the even and odd bytes in the output for every block. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *a, uint16_t *out, int n) {
    for (int i = 0; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_VectorPair p = Q6_Wuh_vzxt_Vub(va);
        *(HVX_Vector *)(out + i)      = Q6_V_hi_W(p);   /* WRONG: swapped */
        *(HVX_Vector *)(out + i + 64) = Q6_V_lo_W(p);   /* WRONG: swapped */
    }
}
