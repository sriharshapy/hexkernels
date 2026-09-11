#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Single-head int8 attention BLOCK with ROTARY POSITION EMBEDDING (RoPE) applied
 * to Q and K before the QK^T matmul -- the marquee L3 kernel i8_attention_block
 * extended with a per-position RoPE pre-rotation:
 *
 *   RoPE (per query/key row i, per feature pair (d, d+HALF), HALF = D/2):
 *     c = cos_lut[i*HALF + d]   (int8, runtime table, Q3 amplitude ~8 = "1.0")
 *     s = sin_lut[i*HALF + d]   (int8, runtime table)
 *     Qr[i*D+d]      = clamp_u8( (Q[i*D+d]*c - Q[i*D+d+HALF]*s) >> rot_shift )
 *     Qr[i*D+d+HALF] = clamp_u8( (Q[i*D+d]*s + Q[i*D+d+HALF]*c) >> rot_shift )
 *     Kr[i*D+d]      = clamp_i8( (K[i*D+d]*c - K[i*D+d+HALF]*s) >> rot_shift )
 *     Kr[i*D+d+HALF] = clamp_i8( (K[i*D+d]*s + K[i*D+d+HALF]*c) >> rot_shift )
 *   (Qr is the uint8 HMX activation -> clamped to [0,255]; Kr is the int8 weight.
 *    The >> is an arithmetic shift; the rotation is the standard fixed-point RoPE
 *    quantization -- the tables carry a built-in 1/16 scale via rot_shift.)
 *
 *   scores[i][j] = sum_d Qr[i*D+d] * Kr[j*D+d]           (rotated Q.K^T, K row j is key j)
 *   s12[i][j]    = ((scores*17 + 8) >> 4) & 0xFFF          (HMX 0x40-config requant field)
 *   scaled[i][j] = clamp( sx12(s12) >> 4 , -128, 127)      (attention SCALE, ~1/sqrt(D) regime)
 *   probs[i][:]  = softmax_lut( scaled[i][:] ) over the KEY axis j   (uint8, sum ~255)
 *                    m = max_j scaled[i][j]
 *                    diff = clamp(scaled[i][j]-m, -255, 0); e = exp_lut[diff+255]
 *                    probs[i][j] = (e*255 + (S_r/2)) / S_r      (S_r = sum_j e)
 *   out_raw[i][d]= sum_j probs[i][j] * V[j*D+d]                  (probs uint8, V int8)
 *   out[i*D+d]   = ((out_raw*17 + 8) >> 4) & 0xFFF               (HMX 0x40-config requant)
 *
 * RoPE runs on HVX/scalar; QK^T and A.V run on the HMX matrix engine (crouton-packed,
 * VTCM-resident tiles); scale+softmax runs on HVX between them. The whole block is
 * FIXED-POINT integer, so the output is BIT-EXACT to the scalar reference (the LUT
 * makes the softmax exact -- no tolerance needed).
 *
 * S=64, D=64, HALF=32. Each matmul is a 2x2 grid of 32x32 crouton tiles with a
 * 2-tile reduction. The harness enables the HMX context before calling you and passes
 * runtime cos_lut/sin_lut ([S x HALF] int8) + exp_lut (256 uint8, index 255 = exp(0))
 * + rot_shift. Use VTCM scratch at HVX_VTCM_BASE. Input ranges + rot_shift keep every
 * requant field < 2048 (12-bit field exact, whole block bit-exact).
 */
void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                      const int8_t *cos_lut, const int8_t *sin_lut,
                      uint16_t *out, int S, int D, int rot_shift,
                      const uint8_t *exp_lut);
#endif
