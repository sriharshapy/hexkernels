/* Scalar int8 32x64 matmul (bit-exact) — DENOMINATOR baseline. Plain
 * triple-loop accumulate over all 64 output columns + 0x40 requant. The HMX
 * expert (two N-tiles) must beat this by >=1.2x. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out,
                       int m, int ncol, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < ncol; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) acc += (int)A[i*k+kk] * (int)B[kk*ncol+j];
            out[i*ncol + j] = sx12((acc * 17 + 8) >> 4);
        }
    }
}
