/* Scalar int8 1x1 conv (= In*W^T) + per-channel bias (bit-exact) — DENOMINATOR
 * baseline. Plain triple-loop channel contraction (no vectorization); each
 * cell is In[p][:] . W[co][:] via a scalar accumulate, then the 0x40 requant
 * + bias[co]. The HMX expert must beat this by >=1.2x. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *In, const int8_t *W, const int32_t *bias,
                       int32_t *out, int n) {
    for (int p = 0; p < n; p++) {
        for (int co = 0; co < n; co++) {
            int acc = 0;
            for (int ci = 0; ci < n; ci++) acc += (int)In[p*n+ci] * (int)W[co*n+ci];
            out[p*n + co] = sx12((acc * 17 + 8) >> 4) + bias[co];
        }
    }
}
