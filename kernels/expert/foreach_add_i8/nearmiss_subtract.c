/* Near-miss: subtracts b from a instead of adding.
 * Fails whenever b[t*L+i] != 0, which is common in seeded data. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int T, int L) {
    for (int t = 0; t < T; t++) {
        int base = t * L;
        for (int i = 0; i < L; i++)
            out[base + i] = (int8_t)((int)a[base + i] - (int)b[base + i]);  /* WRONG: minus */
    }
}
