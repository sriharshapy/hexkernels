#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Single-head int8 attention BLOCK (the marquee L3 kernel):
 *
 *   scores[i][j] = sum_d Q[i*D+d] * K[j*D+d]         (Q uint8, K int8; K row j IS
 *                                                     key vector j, so this is Q.K^T
 *                                                     with no explicit transpose)
 *   s12[i][j]    = ((scores*17 + 8) >> 4) & 0xFFF     (HMX 0x40-config requant field)
 *   scaled[i][j] = clamp( sx12(s12) >> 4 , -128, 127) (the attention SCALE -- a
 *                                                     power-of-two shift, net ~1/15,
 *                                                     in the 1/sqrt(D) regime)
 *   probs[i][:]  = softmax_lut(scaled[i][:]) over the KEY axis j   (uint8, sum ~255)
 *                    m = max_j scaled[i][j]
 *                    diff = clamp(scaled[i][j]-m, -255, 0); e = exp_lut[diff+255]
 *                    probs[i][j] = (e*255 + (S_r/2)) / S_r      (S_r = sum_j e)
 *   out_raw[i][d]= sum_j probs[i][j] * V[j*D+d]                  (probs uint8, V int8)
 *   out[i*D+d]   = ((out_raw*17 + 8) >> 4) & 0xFFF               (HMX 0x40-config requant)
 *
 * QK^T and A.V run on the HMX matrix engine (crouton-packed, VTCM-resident tiles);
 * the scale+softmax runs on HVX/scalar between them. The whole block is FIXED-POINT
 * integer, so the block output is BIT-EXACT to the scalar reference (no float, no
 * tolerance needed -- the LUT makes the "float-ish" softmax exact).
 *
 * S=64, D=64: each matmul is a 2x2 grid of 32x32 crouton tiles with a 2-tile
 * reduction. The harness enables the HMX context before calling you and passes a
 * runtime exp_lut (256 uint8 entries, index 255 = exp(0)); use VTCM scratch at
 * HVX_VTCM_BASE. Input ranges are chosen so every requant field is exact (<2048).
 */
void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                      uint16_t *out, int S, int D, const uint8_t *exp_lut);
#endif
