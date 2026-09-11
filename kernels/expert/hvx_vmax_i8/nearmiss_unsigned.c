/* NEAR-MISS: uses the UNSIGNED-byte max (Q6_Vub_vmax_VubVub) instead of the
 * signed int8 max. Compiles and passes when both operands have the same
 * sign, but fails whenever a[i] and b[i] straddle zero (e.g. a=-1,b=1: the
 * signed max is 1, but unsigned-byte max treats -1 as 0xFF=255 and wrongly
 * returns -1). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vub_vmax_VubVub(va, vb);   /* WRONG: unsigned compare */
    }
    for (; i < n; i++) {
        uint8_t x = (uint8_t)a[i], y = (uint8_t)b[i];
        out[i] = (int8_t)((x > y) ? x : y);   /* WRONG: unsigned compare */
    }
}
