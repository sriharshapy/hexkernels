/* NEARMISS: transposes the interleave order — writes out[p*4+c] = in[c*HW+p]
 * but with channels reversed (c -> 3-c). Plausible axis confusion; must score
 * INCORRECT. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int C, int HW) {
    (void)C;
    for (int p=0;p<HW;p++)
        for (int c=0;c<4;c++)
            out[p*4 + c] = in[(3-c)*HW + p];   /* reversed channel order */
}
