/* Baseline: scalar per-row depad copy reading/writing DDR directly (no HVX,
 * no VTCM staging, no descriptor chaining). This is the speedup denominator:
 * for each row, byte-copy W bytes from the padded source row to the compact
 * destination row. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int8_t *out, int H, int W, int rowstride) {
    for (int r = 0; r < H; r++) {
        const int8_t *src = a + (long)r * rowstride;
        int8_t *dst = out + (long)r * W;
        for (int c = 0; c < W; c++) dst[c] = src[c];
    }
}
