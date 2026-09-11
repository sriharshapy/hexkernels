/* Baseline: scalar requant + branch clamp to signed int8. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const int32_t *acc, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int r = (acc[i] * 17 + 8) >> 4;
        if (r > 127) r = 127; else if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
