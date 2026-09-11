/* NEAR-MISS: uses the FLOOR (truncating) average Q6_Vub_vavg_VubVub instead
 * of the rounding Q6_Vub_vavg_VubVub_rnd. Compiles and passes whenever a+b
 * is even, but fails whenever a+b is odd (off by one low). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vub_vavg_VubVub(va, vb);   /* WRONG: floors, no +1 */
    }
    for (; i < n; i++)
        out[i] = (uint8_t)(((int)a[i] + (int)b[i]) >> 1);   /* WRONG: no rounding */
}
