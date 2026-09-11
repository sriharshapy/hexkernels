/* Baseline: scalar rectangular matmul (M x N, contraction K) + HMX 0x40 requant. */
#include "kernel_api.h"
#include "harness_common.h"

static inline int sx12(int f) { int v = f & 0xFFF; if (v & 0x800) v -= 0x1000; return v; }

void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out,
                       int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int acc = 0;
            for (int k = 0; k < K; k++) acc += (int)A[i*K+k] * (int)B[k*N+j];
            out[i*N + j] = sx12((acc * 17 + 8) >> 4);
        }
}
