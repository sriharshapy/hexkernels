/* Near-miss: applies the requant but NARROWS to int8 by truncation (no
 * saturation). For |r| > 127 the cast wraps modulo 256 instead of clamping ->
 * must FAIL bit-exact (the input range makes r exceed the int8 range often). */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const int32_t *acc, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int r = (acc[i] * 17 + 8) >> 4;
        out[i] = (int8_t)r;   /* truncating cast -- no saturation */
    }
}
