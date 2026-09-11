/* Scalar int8 matmul with K=64 reduction (bit-exact) — DENOMINATOR baseline.
 * Plain triple-loop accumulate over the full K=k_dim, then the 0x40 requant.
 * The HMX expert (chained accumulation over two K-tiles) must beat this by
 * >=1.2x. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n, int k_dim) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < k_dim; k++) acc += (int)A[i*k_dim+k] * (int)B[k*n+j];
            out[i*n + j] = sx12((acc * 17 + 8) >> 4);
        }
    }
}
