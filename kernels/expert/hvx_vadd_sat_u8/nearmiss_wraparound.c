/* NEAR-MISS: uses the WRAPPING add (Q6_Vb_vadd_VbVb, bit-identical to a
 * plain mod-256 add) instead of the saturating add. Compiles and passes for
 * non-overflowing inputs, but fails on the pinned saturation edge cases
 * (e.g. 200+200 should saturate to 255, not wrap to 144). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vb_vadd_VbVb(va, vb);   /* WRONG: wraps, doesn't saturate */
    }
    for (; i < n; i++)
        out[i] = (uint8_t)((int)a[i] + (int)b[i]);   /* WRONG: wraps mod 256 */
}
