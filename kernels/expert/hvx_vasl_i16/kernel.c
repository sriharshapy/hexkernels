/* expert: HVX shift-left via Q6_Vh_vasl_VhR (plain wraparound, no sat). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, int16_t *out, int n, int shift) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        *(HVX_Vector *)(out + i) = Q6_Vh_vasl_VhR(va, shift);
    }
    for (; i < n; i++)
        out[i] = (int16_t)((uint16_t)a[i] << shift);
}
