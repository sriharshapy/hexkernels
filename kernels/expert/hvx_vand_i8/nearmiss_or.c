/* NEAR-MISS: uses bitwise OR (Q6_V_vor_VV) instead of AND. Compiles and
 * passes only when a[i]==b[i] for every element (OR==AND in that case);
 * fails whenever the operands' bit patterns differ. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_V_vor_VV(va, vb);   /* WRONG: OR, not AND */
    }
    for (; i < n; i++)
        out[i] = (int8_t)((uint8_t)a[i] | (uint8_t)b[i]);   /* WRONG: OR, not AND */
}
