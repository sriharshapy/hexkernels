#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* L3 int8 TRANSFORMER ENCODER BLOCK -- the marquee composite. Composes the
 * i8_attention_block and i8_ffn_block pipelines with two residual adds, all int8,
 * all four matmuls on the HMX matrix engine, VTCM-staged, HVX epilogues:
 *
 *   Self-attention sub-layer (Q = K = V = X, identity QKV projections):
 *     scores[i][j] = sum_d X[i*D+d] * X[j*D+d]                (X.X^T, HMX)
 *     s12          = ((scores*17+8)>>4) & 0xFFF                (0x40 requant field)
 *     scaled[i][j] = clamp( sx12(s12) >> ATTN_SCALE_SHIFT , -128, 127)
 *     probs[i][:]  = softmax_lut(scaled[i][:]) over the KEY axis j (uint8, sum ~255)
 *     av[i][d]     = sum_j probs[i][j] * X[j*D+d]              (A.V, HMX)
 *     avf          = ((av*17+8)>>4) & 0xFFF
 *     a[i][d]      = clamp( sx12(avf) >> ATTN_OUT_SHIFT , -128, 127)  (int8 attn out)
 *
 *   Residual add 1 (attn output + input):
 *     h[i][d]      = clamp_u8( X[i*D+d] + a[i*D+d] )           (uint8, FFN activation)
 *
 *   Feed-forward sub-layer (up-proj -> ReLU+requant -> down-proj):
 *     acc1[i][j]   = sum_k h[i*D+k] * W1[k*Dff+j]              (HMX)
 *     r1           = sx12((acc1*17+8)>>4)
 *     H[i][j]      = (r1+b1[j] > 0) ? clamp((r1+b1[j]) >> FFN_SH1, 0, 127) : 0
 *     acc2[i][j]   = sum_k H[i*Dff+k] * W2[k*D+j]              (HMX)
 *     r2           = sx12((acc2*17+8)>>4)
 *     f[i][j]      = sat_i8( (r2+b2[j]) >> FFN_SH2 )           (int8 FFN out)
 *
 *   Residual add 2 (FFN output + residual stream):
 *     out[i][d]    = sat_i8( h[i*D+d] + f[i*D+d] )             (int8 block output)
 *
 * Documented simplifications (fit the bare-sim budget + the 12-bit HMX requant field
 * chained across four matmuls): self-attention uses X directly as Q,K,V (no separate
 * QKV/output projections); LayerNorm is omitted in favour of a residual+requant path
 * (norms would blow the sim budget and add no new mechanism). Input ranges + the shift
 * constants below keep every requant field < 2048, so the whole block is BIT-EXACT.
 *
 * S=D=64, Dff=128 (all 32-multiples for the crouton tile edge). The harness enables the
 * HMX context AND installs an identity VTCM translation before the timed call. Use VTCM
 * scratch at HVX_VTCM_BASE. exp_lut is 256 uint8 (runtime, index 255 = exp(0)). */

#define ATTN_SCALE_SHIFT 4   /* attention internal 1/sqrt(D)-regime scale */
#define ATTN_OUT_SHIFT   8   /* attention output requant -> int8          */
#define FFN_SH1          8   /* FFN intermediate ReLU-requant shift        */
#define FFN_SH2          4   /* FFN output requant shift                   */

void candidate_kernel(const uint8_t *X,
                      const int8_t *W1, const int32_t *b1,   /* [D x Dff], [Dff] */
                      const int8_t *W2, const int32_t *b2,   /* [Dff x D], [D]   */
                      const uint8_t *exp_lut,                /* [256] uint8      */
                      int8_t *out, int S, int D, int Dff);

static inline int tf_sx12(int field) {   /* sign-extend HMX 12-bit requant field */
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}
static inline int tf_clamp_i8(int v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return v;
}
static inline int tf_clamp_u8(int v) {
    if (v > 255) return 255;
    if (v <   0) return 0;
    return v;
}
static inline int tf_relu_requant(int p1) {   /* ReLU + shift -> int8 activation [0,127] */
    int hh = (p1 > 0) ? (p1 >> FFN_SH1) : 0;
    if (hh > 127) hh = 127;
    return hh;
}
static inline signed char tf_sat_i8(int v) {
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (signed char)v;
}
#endif
