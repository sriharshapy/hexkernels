/* NEAR-MISS: uses the SATURATING add (Q6_Vb_vadd_VbVb_sat) instead of the
 * wrapping add. Compiles and passes for non-overflowing inputs, but fails
 * bit-exact on the pinned overflow edge cases (e.g. 127+127 should wrap to
 * -2, not saturate to 127). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vb_vadd_VbVb_sat(va, vb);   /* WRONG: saturating */
    }
    for (; i < n; i++) {
        int r = (int)a[i] + (int)b[i];
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
