/* NEAR-MISS: treats the bytes as SIGNED before differencing (uses
 * Q6_Vb_vabsdiff-style signed logic emulated in scalar, and a plain byte
 * subtract that wraps in HVX rather than the correct wide-unsigned diff).
 * Compiles and passes when both operands are < 128, but fails once
 * either has the high bit set (e.g. |0-255| should be 255, not 1). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int d = (int)(int8_t)a[i] - (int)(int8_t)b[i];   /* WRONG: signed reinterpret */
        out[i] = (uint8_t)(d < 0 ? -d : d);
    }
}
