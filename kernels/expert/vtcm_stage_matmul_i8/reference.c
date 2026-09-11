/* Baseline: naive scalar GEMM (triple loop, int64 accumulate cast to int32).
 * Speedup denominator: correct but no vectorization, no packing, no VTCM
 * staging -- pays full DDR latency and scalar throughput on the large A
 * operand streamed row by row. */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *X, int32_t *C,
                      int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int j = 0; j < N; j++) {
            int64_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int64_t)A[(long)m * K + k] * (int64_t)X[(long)k * N + j];
            C[(long)m * N + j] = (int32_t)acc;
        }
    }
}
