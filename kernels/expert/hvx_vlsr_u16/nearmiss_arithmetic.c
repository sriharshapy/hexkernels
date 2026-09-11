/* NEAR-MISS: uses an ARITHMETIC (sign-extending) shift-right by
 * reinterpreting the input as signed, instead of the correct logical
 * (zero-fill) shift. Compiles and passes when bit 15 is clear, but fails
 * bit-exact whenever a[i] >= 0x8000 (e.g. 0x8000>>5 should give 1024, not
 * 0xFC00=64512-as-unsigned... actually wraps to 0xFC00). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint16_t *a, uint16_t *out, int n, int shift) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        *(HVX_Vector *)(out + i) = Q6_Vh_vasr_VhR(va, shift);  /* WRONG: arithmetic (signed) */
    }
    for (; i < n; i++) {
        int16_t sv = (int16_t)a[i];
        out[i] = (uint16_t)(int16_t)(sv >> shift);   /* WRONG: sign-extends */
    }
}
