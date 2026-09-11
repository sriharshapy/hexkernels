/* NEAR-MISS: assumes "negative average" means -((a+b)>>1) instead of the
 * actual hardware semantics (a-b)>>1. Compiles, and happens to match when
 * b==-b (e.g. b=0), but fails bit-exact on generic inputs (e.g. a=10,b=4:
 * correct is 3, this gives -7). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int8_t)(-(((int)a[i] + (int)b[i]) >> 1));   /* WRONG formula */
}
