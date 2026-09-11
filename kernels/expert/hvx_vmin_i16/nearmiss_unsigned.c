/* NEAR-MISS: uses the UNSIGNED-halfword min (Q6_Vuh_vmin_VuhVuh) instead of
 * the signed int16 min. Compiles and passes when both operands have the
 * same sign, but fails whenever a[i] and b[i] straddle zero (e.g. a=-1,b=1:
 * signed min is -1, but unsigned-halfword min treats -1 as 0xFFFF=65535 and
 * wrongly returns 1). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vuh_vmin_VuhVuh(va, vb);   /* WRONG: unsigned compare */
    }
    for (; i < n; i++) {
        uint16_t x = (uint16_t)a[i], y = (uint16_t)b[i];
        out[i] = (int16_t)((x < y) ? x : y);   /* WRONG: unsigned compare */
    }
}
