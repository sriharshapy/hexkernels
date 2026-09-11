#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * MLP-Mixer token-mixing block (integer, no float):
 *
 * Input X: [TOKENS x CHANNELS] int8
 * Weight W: [D_MIX x TOKENS] int8  (token-mixing MLP weight)
 * Bias b: [D_MIX] int32
 * Weight W2: [TOKENS x D_MIX] int8  (projection back to token dim)
 * Bias b2: [TOKENS] int32
 * Output: [TOKENS x CHANNELS] int8
 *
 * Algorithm (per channel c):
 *   Step 1: token-mix GEMM:
 *     For each mixer row m in [0, D_MIX):
 *       y1[m] = sat_i8(round_half_away( (sum_t W[m,t]*X[t,c] + b[m]) * mult1, shift1 ))
 *
 *   Step 2: project back:
 *     For each token t in [0, TOKENS):
 *       out[t,c] = sat_i8(round_half_away( (sum_m W2[t,m]*y1[m] + b2[t]) * mult2, shift2 ))
 *
 * This is repeated independently for each channel c.
 * (Token-mixing acts on the token axis while channel-mixing acts on channel axis.)
 *
 * Layout:
 *   X:   [TOKENS x CHANNELS] int8 row-major
 *   W:   [D_MIX x TOKENS]    int8 row-major
 *   b:   [D_MIX]             int32
 *   W2:  [TOKENS x D_MIX]    int8 row-major
 *   b2:  [TOKENS]            int32
 *   out: [TOKENS x CHANNELS] int8 row-major
 *
 * All params are runtime; do NOT hardcode.
 * TOKENS=8, CHANNELS=16, D_MIX=16
 */
#define TOKENS   8
#define CHANNELS 16
#define D_MIX    16

void candidate_kernel(const int8_t  *X,      /* [TOKENS x CHANNELS] */
                      const int8_t  *W,      /* [D_MIX x TOKENS] */
                      const int32_t *b,      /* [D_MIX] */
                      const int8_t  *W2,     /* [TOKENS x D_MIX] */
                      const int32_t *b2,     /* [TOKENS] */
                      int8_t        *out,    /* [TOKENS x CHANNELS] */
                      int32_t mult1, int shift1,
                      int32_t mult2, int shift2);
#endif /* KERNEL_API_H */
