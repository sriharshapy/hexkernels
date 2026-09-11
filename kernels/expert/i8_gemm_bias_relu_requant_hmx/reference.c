/* Pure scalar 64x64 GEMM + bias + ReLU + saturating requant-to-uint8 (bit-exact)
 * — the DENOMINATOR baseline. No HMX/HVX; plain nested-loop matmul, then the
 * same HMX-0x40-requant -> bias -> ReLU -> saturate epilogue as the expert,
 * applied per output element. */
#include "kernel_api.h"
#include <stdint.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

static inline uint8_t saturate_u8(int v) {
    if (v > 255) v = 255;
    if (v <   0) v = 0;
    return (uint8_t)v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias,
                       uint8_t *out, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++) acc += (int)A[i*n+k] * (int)B[k*n+j];
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[j];
            int relu = biased > 0 ? biased : 0;
            out[i*n + j] = saturate_u8(relu);
        }
}
