/* Near-miss: two's-complement wraparound add instead of saturating add.
 * 127+1 wraps to -128 instead of saturating at 127. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    const int vlen = sizeof(HVX_Vector);
    int i = 0;
    for (; i + vlen <= n; i += vlen)
        *(HVX_Vector *)(out + i) = Q6_Vb_vadd_VbVb(*(const HVX_Vector *)(a + i), *(const HVX_Vector *)(b + i));
    for (; i < n; i++) out[i] = (int8_t)(a[i] + b[i]);
}
