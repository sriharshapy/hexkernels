/* Baseline: simple scalar K=128 deep-K matmul + HMX 0x40 requant field + ReLU. */
#include "kernel_api.h"
#include "harness_common.h"

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n, int K) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < K; k++) acc += (int)A[i*K+k] * (int)B[k*n+j];
            int r = sx12((acc * 17 + 8) >> 4);
            out[i*n + j] = r > 0 ? r : 0;
        }
}
