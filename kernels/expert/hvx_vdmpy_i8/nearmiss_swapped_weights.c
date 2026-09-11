/* NEAR-MISS: swaps the weight pairing within each group -- pairs byte 0
 * with w[1] and byte 1 with w[0] (and similarly for bytes 2,3 / w[2],w[3])
 * instead of the correct w[0]/w[1] and w[2]/w[3] order. Compiles and
 * happens to match only when w[0]==w[1] or a[4k]==a[4k+1] (never, with our
 * distinct fixed weights), so it fails on essentially every group. */
#include <stdint.h>

void candidate_kernel(const uint8_t *a, const int8_t w[4], int16_t *out, int g) {
    for (int k = 0; k < g; k++) {
        int s0 = (int)a[4*k+0]*w[1] + (int)a[4*k+1]*w[0];   /* WRONG: swapped */
        int s1 = (int)a[4*k+2]*w[3] + (int)a[4*k+3]*w[2];   /* WRONG: swapped */
        out[2*k]   = (int16_t)s0;
        out[2*k+1] = (int16_t)s1;
    }
}
