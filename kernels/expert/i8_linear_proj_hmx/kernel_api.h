#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 linear projection (the attention QKV / output projection) on the HMX
 * matrix engine, with a fused per-output-channel bias epilogue.
 *
 *   X: [S x Din]  uint8, row-major (activations, S=64 tokens).
 *   W: [Dout x Din] int8, row-major (nn.Linear-style weight layout --
 *                                     row o IS output-channel o's weight
 *                                     vector, Din=Dout=64).
 *   bias: [Dout] int32.
 *   out: [S x Dout] int8 (bit-exact, saturating narrow -- NO activation,
 *                          this is a plain projection).
 *
 * Pinned formula:
 *   acc[i][o] = sum_d X[i*Din+d] * W[o*Din+d]           (int32)
 *   r[i][o]   = sign_extend_12bit((acc*17 + 8) >> 4)     (0x40-config HMX requant field)
 *   biased    = r[i][o] + bias[o]                         (int32, per-output-channel bias)
 *   out[i][o] = saturate_i8(biased)                        (clamp to [-128,127])
 *
 * S=64 is a 2x2 grid of 32x32 output tiles; each output tile accumulates over
 * two Din-tiles (Din=64 = 2*32). The harness enables the HMX context before
 * calling you; use VTCM scratch at HVX_VTCM_BASE. Input ranges are chosen so
 * |requant result| < 2048 (the 12-bit field is exact). */
void candidate_kernel(const uint8_t *X, const int8_t *W, const int32_t *bias,
                       int8_t *out, int n);
#endif
