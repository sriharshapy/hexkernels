/* NEAR-MISS: uses a control vector of 0xFE (bit 0 clear) instead of 0xFF.
 * vdelta's swap network needs bits [6:0] set to drive a full reversal of
 * a 128-lane vector (2^7=128); clearing bit 0 disables the finest-grain
 * (distance-1) swap stage, so adjacent byte pairs are NOT swapped with
 * each other -- the result is a "reverse by pairs", not a true full
 * byte reversal. Compiles fine but wrong for any block whose adjacent
 * bytes differ. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    HVX_Vector vctrl = Q6_Vb_vsplat_R(0xFE);   /* WRONG: bit0 cleared */
    for (int i = 0; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        *(HVX_Vector *)(out + i) = Q6_V_vdelta_VV(va, vctrl);
    }
}
