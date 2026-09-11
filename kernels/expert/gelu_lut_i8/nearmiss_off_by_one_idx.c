/* Plausible-but-WRONG: off-by-one shift in the index (indexes with x[i]+1
 * instead of x[i]). Wrong on essentially every element except where the LUT
 * happens to be locally flat (GELU is nearly flat only deep in the negative
 * saturation region -- most of the domain, including the pinned edge values
 * and the random inputs, will diverge). */
#include "kernel_api.h"
#include <stdint.h>

void candidate_kernel(const int8_t *x, int8_t *out, int n, const int8_t *lut) {
    for (int i = 0; i < n; i++) {
        uint8_t idx = (uint8_t)(x[i] + 1);  /* BUG: off-by-one index shift */
        out[i] = lut[idx];
    }
}
