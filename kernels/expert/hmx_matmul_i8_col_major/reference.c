/* Scalar int8 matmul, column-major output (bit-exact) — DENOMINATOR baseline.
 * Plain triple-loop accumulate + 0x40 requant; store the (i,j) result
 * column-major to out[j*n+i]. The HMX expert must beat this by >=1.2x. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++) acc += (int)A[i*n+k] * (int)B[k*n+j];
            out[j*n + i] = sx12((acc * 17 + 8) >> 4);   /* column-major */
        }
    }
}
