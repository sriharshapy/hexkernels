#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* L3 int8 MLP-MIXER BLOCK -- TWO transposed matmul stages, each its own 2-layer
 * MLP + residual: TOKEN-mixing (mixes across the SEQUENCE/TOKEN axis, weights
 * shared across channels) THEN CHANNEL-mixing (mixes across the FEATURE axis,
 * weights shared across tokens). Pure HVX (no HMX), fixed-point integer ->
 * BIT-EXACT.
 *
 *   --- Token-mixing MLP (mixes over the S axis; same Wt1/Wt2 for every channel d) ---
 *   for d in [0,D):
 *     for s' in [0,St):
 *       acc1[s'][d] = sum_{s=0}^{S-1} X[s*D+d] * Wt1[s'*S+s]
 *       TMh[s'][d]  = relu_i8( (acc1[s'][d] >> TM_SH1) + bt1[s'] )     (uint8, 0..127)
 *     for s in [0,S):
 *       acc2[s][d]  = sum_{s'=0}^{St-1} TMh[s'][d] * Wt2[s*St+s']
 *       TMo[s][d]   = sat_i8( (acc2[s][d] >> TM_SH2) + bt2[s] )
 *   Y[s][d] = sat_i8( X[s*D+d] + TMo[s][d] )                           (residual 1)
 *
 *   --- Channel-mixing MLP (mixes over the D axis; same Wc1/Wc2 for every token s) ---
 *   for s in [0,S):
 *     for j in [0,Dff):
 *       acc1[s][j] = sum_{d=0}^{D-1} Y[s*D+d] * Wc1[d*Dff+j]
 *       CMh[s][j]  = relu_i8( (acc1[s][j] >> CM_SH1) + bc1[j] )        (uint8, 0..127)
 *     for d in [0,D):
 *       acc2[s][d] = sum_{j=0}^{Dff-1} CMh[s][j] * Wc2[j*D+d]
 *       CMo[s][d]  = sat_i8( (acc2[s][d] >> CM_SH2) + bc2[d] )
 *   out[s][d] = sat_i8( Y[s*D+d] + CMo[s][d] )                          (residual 2)
 *
 * relu_i8(p) = (p>0) ? clamp(p, 0, 127) : 0
 *
 * S=32 (tokens), D=64 (channels), St=64 (token-mix hidden), Dff=64 (channel-mix
 * hidden). X uint8 (0..3, input activations); Wt1/Wc1 int8 (-3..3); Wt2/Wc2 int8
 * (-2..2); biases int32. Everything is plain integer arithmetic (fixed shifts,
 * no HMX 12-bit requant field), so bit-exactness only requires summing the SAME
 * terms in ANY order (integer add/mul is associative). The token-mixing stage
 * operates on X's TRANSPOSE (mixing tokens, weights shared across channels);
 * the channel-mixing stage operates on Y directly (mixing channels, weights
 * shared across tokens) -- these are two genuinely different reduction axes
 * over the SAME tensor, the defining MLP-Mixer structure.
 */

#define TM_SH1 2   /* token-mix hidden accumulator -> uint8 activation shift  */
#define TM_SH2 7   /* token-mix output accumulator -> int8 shift              */
#define CM_SH1 8   /* channel-mix hidden accumulator -> uint8 activation shift */
#define CM_SH2 7   /* channel-mix output accumulator -> int8 shift            */

void candidate_kernel(const uint8_t *X,
                      const int8_t *Wt1, const int32_t *bt1,   /* [St x S], [St] */
                      const int8_t *Wt2, const int32_t *bt2,   /* [S x St], [S]  */
                      const int8_t *Wc1, const int32_t *bc1,   /* [D x Dff], [Dff] */
                      const int8_t *Wc2, const int32_t *bc2,   /* [Dff x D], [D]   */
                      int8_t *out, int S, int D, int St, int Dff);

static inline int mx_clamp_i8(int v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return v;
}
static inline signed char mx_sat_i8(int v) {
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (signed char)v;
}
static inline int mx_relu_i8(int p) {   /* ReLU + clamp -> uint8 activation [0,127] */
    if (p <= 0) return 0;
    return (p > 127) ? 127 : p;
}
#endif
