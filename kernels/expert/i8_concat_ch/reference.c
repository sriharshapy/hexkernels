#include <stdint.h>
/*
 * Channel concat baseline.
 * First C1 channels from a, then C2 channels from b.
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out,
                      int C1, int C2, int H, int W) {
    int plane = H * W;
    /* Copy a's channels */
    for (int c = 0; c < C1; c++)
        for (int i = 0; i < plane; i++)
            out[c * plane + i] = a[c * plane + i];
    /* Copy b's channels starting at offset C1 */
    for (int c = 0; c < C2; c++)
        for (int i = 0; i < plane; i++)
            out[(C1 + c) * plane + i] = b[c * plane + i];
}
