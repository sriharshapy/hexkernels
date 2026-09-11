/* NEAR-MISS: computes the Q15 multiply-high WITHOUT the rounding bias
 * (plain truncating divide by 32768), instead of the correct
 * round-half-away-from-zero. Compiles and matches when P is an exact
 * multiple of 32768, but fails bit-exact on generic inputs (e.g.
 * a=100,b=200: correct rounds to 1, plain truncation gives 0). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int32_t P = (int32_t)a[i] * (int32_t)b[i];
        int32_t q = P / 32768;   /* WRONG: no rounding bias */
        if (q > 32767) q = 32767;
        if (q < -32768) q = -32768;
        out[i] = (int16_t)q;
    }
}
