/* NEAR-MISS: uses the SATURATING abs (Q6_Vh_vabs_Vh_sat) instead of the
 * wrapping abs. Compiles and passes for every input except INT16_MIN, where
 * it saturates to 32767 instead of wrapping to -32768. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, int16_t *out, int n) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        *(HVX_Vector *)(out + i) = Q6_Vh_vabs_Vh_sat(va);   /* WRONG: saturating */
    }
    for (; i < n; i++) {
        int v = a[i];
        int r = v < 0 ? -v : v;
        if (r > 32767) r = 32767;   /* WRONG: saturates instead of wrapping */
        out[i] = (int16_t)r;
    }
}
