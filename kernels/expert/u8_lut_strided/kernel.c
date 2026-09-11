/* u8_lut_strided sol_05: memcpy bulk then overwrite LUT positions
   Uses HVX for bulk copy, scalar for LUT overwrites */
#include "kernel_api.h"
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *in, uint8_t *out, int n,
                      const uint8_t *lut, int k) {
    /* Bulk copy via HVX */
    const int vlen = 128;
    int i = 0;
    for (; i + vlen <= n; i += vlen)
        *(HVX_Vector *)(out + i) = *(const HVX_Vector *)(in + i);
    for (; i < n; i++)
        out[i] = in[i];
    /* Overwrite every k-th position with LUT result */
    for (int j = 0; j < n; j += k)
        out[j] = lut[in[j]];
}