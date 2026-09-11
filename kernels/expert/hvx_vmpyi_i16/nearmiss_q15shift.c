/* NEAR-MISS: a very plausible confusion between the plain integer multiply
 * (vmpyi, low 16 bits of the exact product) and a Q15 FIXED-POINT
 * fractional multiply (which right-shifts the product by 15 before
 * narrowing). Compiles and even matches on small-magnitude inputs where the
 * >>15 rarely changes the truncated low bits by coincidence, but fails on
 * the pinned edge cases (e.g. 300*300 should truncate to 24464, not shift
 * to (90000>>15)=2). */
#include <stdint.h>

void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int16_t)(((int)a[i] * (int)b[i]) >> 15);   /* WRONG: Q15 shift */
}
