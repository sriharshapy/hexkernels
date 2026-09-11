#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Grouped-Query Attention BLOCK (the L3 marquee GQA kernel) -- a full
 * multi-head attention layer where H_Q query heads SHARE one KV head:
 *
 *   [HMX] QK^T  ->  [HVX] scale+softmax  ->  [HMX] A.V     (per query head)
 *
 *   Q:   [H_Q  x S x D] uint8 (0..7), row-major, head outermost.
 *   K,V: [H_KV x S x D] int8  (-3..3), row-major, head outermost. K row j IS
 *        key vector j (K is already K^T-shaped, as in the single-head sibling).
 *   out: [H_Q  x S x D] uint16 (12-bit HMX requant field), head outermost.
 *
 * Per query head h_q, shared KV head h_kv = h_q / GROUP_SIZE:
 *   scores[i][j] = sum_d Q[h_q,i,d] * K[h_kv,j,d]           (Q.K^T)
 *   s12[i][j]    = ((scores*17 + 8) >> 4) & 0xFFF            (HMX 0x40-config requant)
 *   scaled[i][j] = clamp( sx12(s12) >> 4 , -128, 127)        (attention SCALE)
 *   probs[i][:]  = softmax_lut( scaled[i][:] ) over the KEY axis j   (uint8, sum ~255)
 *                    m    = max_j scaled[i][j]
 *                    e_j  = exp_lut[ clamp(scaled[i][j]-m, -255, 0) + 255 ]
 *                    probs[i][j] = (e_j*255 + (sum_j e_j)/2) / (sum_j e_j)
 *   out_raw[i][d]= sum_j probs[i][j] * V[h_kv,j,d]           (probs uint8, V int8)
 *   out[h_q,i,d] = ((out_raw*17 + 8) >> 4) & 0xFFF           (HMX 0x40-config requant)
 *
 * H_Q=2, H_KV=1, GROUP_SIZE=2 (both query heads share the single KV head),
 * S=64, D=64. Both matmuls run on the HMX matrix engine; the scale+softmax runs
 * on HVX between them. The GQA mechanism win: the shared KV head's HMX weight
 * croutons (K for QK^T, V for A.V) are packed into VTCM ONCE and REUSED across the
 * GROUP_SIZE query heads that share it -- only the query/probs activation side is
 * re-packed per head. The whole block is fixed-point integer, so the LUT softmax
 * makes it BIT-EXACT to the scalar reference (no tolerance). The harness enables
 * the HMX context before calling you; use VTCM scratch at HVX_VTCM_BASE. Input
 * ranges keep every requant field < 2048 (12-bit field exact). */
#define GQA_H_Q        2
#define GQA_H_KV       1
#define GQA_GROUP_SIZE 2   /* H_Q / H_KV */
#define GQA_S          64
#define GQA_D          64

void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                      uint16_t *out, const uint8_t *exp_lut);
#endif
