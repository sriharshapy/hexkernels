/* s1: direct byte-splat intrinsic Q6_Vb_vsplat_R. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(int8_t val, int8_t *out, int n) {
    HVX_Vector vsplat = Q6_Vb_vsplat_R((int)val);
    int i = 0;
    for (; i + 128 <= n; i += 128)
        *(HVX_Vector *)(out + i) = vsplat;
    for (; i < n; i++)
        out[i] = val;
}
