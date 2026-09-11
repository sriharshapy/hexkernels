#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* L3 int8 TRANSFORMER DECODER BLOCK -- causal self-attention (masked) + residual +
 * LUT-activation FFN + residual. Distinct from i8_transformer_encoder_block in TWO
 * ways: (1) CAUSAL masking on the attention scores (query i only attends to keys
 * j<=i -- the defining decoder property), and (2) the FFN nonlinearity is a runtime
 * LOOKUP TABLE (act_lut, e.g. GELU-shaped) instead of the encoder's closed-form ReLU
 * formula. Pure HVX (no HMX) -- all fixed-point integer, so the whole block is
 * BIT-EXACT.
 *
 *   Causal self-attention (Q = K = V = X, identity projections):
 *     for i in [0,S):
 *       for j in [0,i]  (CAUSAL: j<=i only; j>i is masked out of the softmax):
 *         raw          = sum_d X[i*D+d] * X[j*D+d]              (X.X^T, uint8xuint8)
 *         scaled[i][j] = clamp_i8( raw >> ATTN_SCALE_SHIFT )
 *       probs[i][0..i] = softmax_lut(scaled[i][0..i])  over the CAUSAL window only
 *                        (uint8, sums to ~255; positions j>i never enter the sum)
 *       for d in [0,D):
 *         av           = sum_{j=0}^{i} probs[i][j] * X[j*D+d]
 *         a[i][d]      = clamp_i8( av >> ATTN_OUT_SHIFT )
 *
 *   Residual add 1:
 *     h[i][d] = clamp_u8( X[i*D+d] + a[i*D+d] )                  (uint8, FFN activation)
 *
 *   Feed-forward sub-layer (up-proj -> LUT activation -> down-proj):
 *     acc1[i][j] = sum_k h[i*D+k] * W1[k*Dff+j]                  (uint8 x int8)
 *     p1         = (acc1 >> FFN_SH1) + b1[j]
 *     idx         = clamp_i8(p1) + 128                            (0..255 LUT index)
 *     H[i][j]    = act_lut[idx]                                  (int8, runtime LUT)
 *     acc2[i][j] = sum_k H[i*Dff+k] * W2[k*D+j]                  (int8 x int8, signed dot)
 *     p2         = (acc2 >> FFN_SH2) + b2[j]
 *     f[i][j]    = sat_i8(p2)
 *
 *   Residual add 2:
 *     out[i][d] = sat_i8( h[i*D+d] + f[i*D+d] )                  (int8 block output)
 *
 * S=32, D=64, Dff=128. softmax_lut uses the same runtime fixed-point exp_lut[256]
 * scheme as i8_transformer_encoder_block (index = 255 - clamp(m-scaled,0,255), where
 * m is the row causal-window max). Do NOT hardcode exp_lut or act_lut -- both are
 * runtime inputs (anti-hardcode). Everything is plain integer arithmetic (no HMX
 * 12-bit requant field), so bit-exactness only requires computing the SAME integer
 * ops in the SAME order of summation (integer add/mul is associative -- vectorizing
 * the dot products does not change the bit-exact result as long as no extra terms
 * beyond the causal window / real reduction length are summed).
 */

#define ATTN_SCALE_SHIFT 2   /* raw X.X^T score -> int8 scaled score           */
#define ATTN_OUT_SHIFT   8   /* attention-weighted-V accumulator -> int8       */
#define FFN_SH1          8   /* FFN up-proj accumulator -> LUT index pre-shift */
#define FFN_SH2          7   /* FFN down-proj accumulator -> int8 output shift */

void candidate_kernel(const uint8_t *X,
                      const int8_t *W1, const int32_t *b1,     /* [D x Dff], [Dff] */
                      const int8_t *act_lut,                    /* [256] int8       */
                      const int8_t *W2, const int32_t *b2,      /* [Dff x D], [D]   */
                      const uint8_t *exp_lut,                   /* [256] uint8      */
                      int8_t *out, int S, int D, int Dff);

static inline int dec_clamp_i8(int v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return v;
}
static inline int dec_clamp_u8(int v) {
    if (v > 255) return 255;
    if (v <   0) return 0;
    return v;
}
static inline signed char dec_sat_i8(int v) {
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (signed char)v;
}
#endif
