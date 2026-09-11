/* Scalar int8 matmul with pre-transposed weight (bit-exact) — DENOMINATOR
 * baseline. Because Bt[j][:] is already the contiguous contraction vector
 * for output column j, each output cell is a plain scalar dot A[i][:] .
 * Bt[j][:], then the same 0x40-config requant. The HMX expert must beat
 * this. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *A, const int8_t *Bt, int32_t *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++) acc += (int)A[i*n+k] * (int)Bt[j*n+k];
            out[i*n + j] = sx12((acc * 17 + 8) >> 4);
        }
    }
}
