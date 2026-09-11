/* NEAR-MISS: saturates the shifted value to int16 range instead of letting
 * it wrap. Compiles and passes when no bits are lost, but fails bit-exact
 * whenever the shift pushes a set bit past bit 15 (e.g. 0x4000<<4 should
 * wrap to 0, not saturate to 32767). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, int16_t *out, int n, int shift) {
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)a[i] << shift;   /* WRONG: wide, then... */
        if (v > 32767) v = 32767;             /* WRONG: saturates */
        if (v < -32768) v = -32768;           /* WRONG: saturates */
        out[i] = (int16_t)v;
    }
}
