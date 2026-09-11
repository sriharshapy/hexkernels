/* Near-miss: wrong offset for b — uses C2 instead of C1 as the channel offset.
   Results in b channels being written to the wrong position (offset by C2-C1). */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out,
                      int C1, int C2, int H, int W) {
    int plane = H * W;
    for (int c = 0; c < C1; c++)
        for (int i = 0; i < plane; i++)
            out[c * plane + i] = a[c * plane + i];
    /* BUG: offset should be C1, not C2 */
    for (int c = 0; c < C2; c++)
        for (int i = 0; i < plane; i++)
            out[(C2 + c) * plane + i] = b[c * plane + i];
}
