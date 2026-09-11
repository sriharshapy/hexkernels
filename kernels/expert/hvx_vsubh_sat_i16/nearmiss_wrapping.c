/* NEAR-MISS: uses the WRAPPING subtract Q6_Vh_vsub_VhVh instead of the
 * saturating Q6_Vh_vsub_VhVh_sat. Compiles and passes for non-overflowing
 * inputs, but fails bit-exact on the pinned overflow edge cases (e.g.
 * 30000-(-10000) should saturate to 32767, not wrap negative). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vh_vsub_VhVh(va, vb);   /* WRONG: wrapping */
    }
    for (; i < n; i++)
        out[i] = (int16_t)((int)a[i] - (int)b[i]);            /* WRONG: wrapping */
}
