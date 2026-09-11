/* sol_03: HVX-vectorized int8 add.
 * Q6_Vb_vadd_VbVb performs wrapping (non-saturating) byte vector add.
 * T*L = 1024 = 8 * 128, so no tail needed.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int T, int L) {
    int total = T * L;  /* 1024 bytes = 8 HVX vectors */
    int i;
    for (i = 0; i + 128 <= total; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vb_vadd_VbVb(va, vb);
    }
    /* scalar tail (not needed for T*L=1024, but correct for general inputs) */
    for (; i < total; i++)
        out[i] = (int8_t)((int)a[i] + (int)b[i]);
}
