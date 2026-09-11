/* NEAR-MISS: uses the SIGNED min Q6_Vb_vmin_VbVb instead of the unsigned
 * Q6_Vub_vmin_VubVub. Compiles and passes when both operands are < 128,
 * but fails bit-exact once either operand's high bit is set (e.g.
 * min(128,127) should be 127 under unsigned compare, not 128). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vb_vmin_VbVb(va, vb);   /* WRONG: signed */
    }
    for (; i < n; i++) {
        int8_t sa = (int8_t)a[i], sb = (int8_t)b[i];
        out[i] = (uint8_t)((sa < sb) ? sa : sb);              /* WRONG: signed */
    }
}
