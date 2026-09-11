/* Pure scalar int8 64x64 matmul + per-column int32 bias add + GELU-via-LUT
 * (bit-exact) — the DENOMINATOR baseline. Naive triple-loop matmul, then the
 * SAME HMX-0x40-requant -> bias -> saturate -> LUT epilogue as the expert,
 * applied per output element. */
#include "kernel_api.h"
#include "harness_common.h"

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

static inline int8_t saturate_i8(int v) {
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int8_t)v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias,
                       const int8_t *gelu_lut, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++) acc += (int)A[i*n+k] * (int)B[k*n+j];
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[j];
            int8_t pre_lut = saturate_i8(biased);
            uint8_t idx = (uint8_t)(pre_lut + 128);
            out[i*n + j] = gelu_lut[idx];
        }
    }
}
