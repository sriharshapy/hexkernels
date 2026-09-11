#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* L3 int8 TRANSFORMER FFN / MLP BLOCK — two matmuls with a gated (ReLU) activation
 * and requant between them, the canonical feed-forward layer of a transformer.
 * Composes THREE mechanism groups:
 *   - HMX matrix engine for BOTH matmuls (32x32 crouton tiles),
 *   - VTCM for the intermediate activation H (kept on-chip between the matmuls,
 *     plus the HMX crouton operand slots), and
 *   - HVX for the fused requant/ReLU/bias epilogues + bulk on-chip data movement.
 *
 * Given X [S x D] (uint8), W1 [D x Dff] (int8), b1 [Dff] (int32),
 *       W2 [Dff x D] (int8), b2 [D] (int32):
 *
 *   acc1[i][j] = sum_{k<D}  X[i*D+k] * W1[k*Dff+j]         (i<S, j<Dff)
 *   r1         = sx12((acc1*17 + 8) >> 4)                   (HMX 0x40-config requant)
 *   p1         = r1 + b1[j]                                  (per-column int32 bias)
 *   H[i][j]    = (p1 > 0) ? clamp(p1 >> FFN_SH1, 0, 127) : 0 (ReLU + requant -> int8)
 *
 *   acc2[i][j] = sum_{k<Dff} H[i*Dff+k] * W2[k*D+j]        (i<S, j<D)
 *   r2         = sx12((acc2*17 + 8) >> 4)
 *   p2         = r2 + b2[j]
 *   out[i][j]  = sat_i8(p2 >> FFN_SH2)                       (int8 output logits)
 *
 * S,D,Dff are multiples of 32 (crouton tile edge). Input ranges keep |r1|,|r2| < 2048
 * so both HMX 12-bit requant fields are exact and the result is BIT-EXACT int8.
 * The harness enables the HMX context AND installs an identity VTCM translation
 * before the timed call, so a candidate may run HMX from VTCM and stage H in VTCM. */
void candidate_kernel(const uint8_t *X, const int8_t *W1, const int32_t *b1,
                      const int8_t *W2, const int32_t *b2,
                      int8_t *out, int S, int D, int Dff);

/* Shared fixed-point pipeline constants (identical across ref/baseline/expert). */
#define FFN_SH1 8   /* intermediate ReLU-requant right shift  */
#define FFN_SH2 3   /* output requant right shift             */

static inline int ffn_sx12(int field) {   /* sign-extend HMX 12-bit requant field */
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}
static inline int ffn_relu_requant(int p1) {   /* ReLU + shift -> int8 activation */
    int h = (p1 > 0) ? (p1 >> FFN_SH1) : 0;
    if (h > 127) h = 127;
    return h;
}
static inline signed char ffn_sat_i8(int v) {
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (signed char)v;
}
#endif
