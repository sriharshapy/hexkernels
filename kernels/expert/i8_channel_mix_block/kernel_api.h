#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * MLP-Mixer channel-mixing block with GELU activation (integer, no float):
 *
 * Input X: [TOKENS x CHANNELS] int8
 * Weight W1: [D_FF x CHANNELS] int8  (expand to D_FF)
 * Bias b1: [D_FF] int32
 * Weight W2: [CHANNELS x D_FF] int8  (project back to CHANNELS)
 * Bias b2: [CHANNELS] int32
 * gelu_lut: 256 int8 entries mapping int8 -> int8 via GELU approximation
 * Output: [TOKENS x CHANNELS] int8
 *
 * Algorithm (per token t):
 *   Step 1: expand GEMM:
 *     For each ff-dim m in [0, D_FF):
 *       acc1  = sum_c W1[m,c]*X[t,c] + b1[m]           (int32)
 *       sat1  = sat_i8(round_half_away(acc1 * mult1, shift1))
 *       y1[m] = gelu_lut[(uint8_t)(sat1 + 128)]         (int8, via LUT)
 *
 *   Step 2: project back GEMM:
 *     For each channel c in [0, CHANNELS):
 *       acc2    = sum_m W2[c,m]*y1[m] + b2[c]           (int32)
 *       out[t,c] = sat_i8(round_half_away(acc2 * mult2, shift2))
 *
 * Layout:
 *   X:        [TOKENS x CHANNELS] int8 row-major
 *   W1:       [D_FF x CHANNELS]   int8 row-major
 *   b1:       [D_FF]              int32
 *   W2:       [CHANNELS x D_FF]   int8 row-major
 *   b2:       [CHANNELS]          int32
 *   gelu_lut: [256]               int8 (runtime, opaque to candidate)
 *   out:      [TOKENS x CHANNELS] int8 row-major
 *
 * All params and tables are RUNTIME; do NOT hardcode.
 * TOKENS=8, CHANNELS=16, D_FF=16
 */
#define TOKENS   8
#define CHANNELS 16
#define D_FF     16

void candidate_kernel(const int8_t  *X,         /* [TOKENS x CHANNELS] */
                      const int8_t  *W1,        /* [D_FF x CHANNELS] */
                      const int32_t *b1,        /* [D_FF] */
                      const int8_t  *W2,        /* [CHANNELS x D_FF] */
                      const int32_t *b2,        /* [CHANNELS] */
                      const int8_t  *gelu_lut,  /* [256] int8 */
                      int8_t        *out,       /* [TOKENS x CHANNELS] */
                      int32_t mult1, int shift1,
                      int32_t mult2, int shift2);
#endif /* KERNEL_API_H */
