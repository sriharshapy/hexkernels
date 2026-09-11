/* Plausible-but-WRONG: mirrors the input around zero before indexing
 * (idx = (uint8_t)(-x[i]) instead of (uint8_t)(x[i])). SiLU is NOT an
 * even/odd-symmetric simple function of -x (unlike, say, tanh which IS
 * odd), so this diverges -- especially far from 0 where the asymmetry is
 * large (the harness pins x=-100 and x=80 specifically to guarantee this is
 * caught; near 0, silu(x) ~= x/2 is nearly odd, which would make a
 * near-zero-only test suite miss this bug). */
#include "kernel_api.h"
#include <stdint.h>

void candidate_kernel(const int8_t *x, int8_t *out, int n, const int8_t *lut) {
    for (int i = 0; i < n; i++) {
        uint8_t idx = (uint8_t)(-x[i]);  /* BUG: mirrored index */
        out[i] = lut[idx];
    }
}
