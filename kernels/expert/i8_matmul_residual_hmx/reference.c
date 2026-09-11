/* Pure scalar int8 64x64 matmul + fused full-matrix int32 residual add
 * (bit-exact) — the DENOMINATOR baseline. Naive triple-loop matmul, then the
 * SAME requant -> elementwise-int32-add epilogue as the expert, applied per
 * output element. */
#include "kernel_api.h"
#include "harness_common.h"

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *C,
                       int32_t *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++) acc += (int)A[i*n+k] * (int)B[k*n+j];
            int r = sx12((acc * 17 + 8) >> 4);
            out[i*n + j] = r + C[i*n + j];
        }
    }
}
