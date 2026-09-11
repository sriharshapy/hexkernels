/* NEAR-MISS: uses the HALFWORD splat Q6_Vh_vsplat_R on the raw (unwidened)
 * int8 value cast straight into a 32-bit register without masking to a
 * single byte -- for negative val this sign-extends into the upper 24
 * bits, and Q6_Vh_vsplat_R replicates the low 16 bits (val plus one byte
 * of sign-extension) instead of the low 8 bits, corrupting every other
 * byte lane for negative inputs. Compiles fine; wrong whenever val < 0. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(int8_t val, int8_t *out, int n) {
    HVX_Vector vsplat = Q6_Vh_vsplat_R((int)val);   /* WRONG: halfword splat */
    int i = 0;
    for (; i + 128 <= n; i += 128)
        *(HVX_Vector *)(out + i) = vsplat;
    for (; i < n; i++)
        out[i] = val;
}
