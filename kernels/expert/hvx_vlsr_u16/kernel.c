/* expert: HVX logical shift-right via Q6_Vuh_vlsr_VuhR. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint16_t *a, uint16_t *out, int n, int shift) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        *(HVX_Vector *)(out + i) = Q6_Vuh_vlsr_VuhR(va, shift);
    }
    for (; i < n; i++)
        out[i] = (uint16_t)(a[i] >> shift);
}
