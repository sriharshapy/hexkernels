#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Grouped-Query Attention QK^T on the HMX matrix engine: H_Q query heads
 * share ONE KV head (GROUP_SIZE = H_Q / H_KV query heads per shared key head).
 *
 *   Q:   [H_Q  x S x D] uint8, row-major, head outermost.
 *   K:   [H_KV x S x D] int8,  row-major, head outermost (row j IS key
 *                                vector j -- K is already K^T-shaped, as in
 *                                the single-head QK^T sibling task).
 *   out: [H_Q  x S x S] uint16 (12-bit HMX requant field).
 *
 * Per query head h_q (shared key head h_kv = h_q / GROUP_SIZE):
 *   scores[h_q][i][j] = sum_d Q[h_q,i,d] * K[h_kv,j,d]
 *   out[h_q][i][j]    = ((scores*17 + 8) >> 4) & 0xFFF   (0x40-config requant)
 *
 * H_Q=2, H_KV=1, GROUP_SIZE=2 (both query heads share the single KV head),
 * S=64, D=64. The mechanism win: the shared KV head's HMX weight croutons
 * are packed into VTCM ONCE and reused (no re-pack) across the GROUP_SIZE
 * query heads that share it -- only the activation (Q) side is re-packed
 * per head/tile. The harness enables the HMX context before calling you; use
 * VTCM scratch at HVX_VTCM_BASE. Input ranges are chosen so
 * |requant result| < 2048 (the 12-bit field is exact). */
#define GQA_H_Q        2
#define GQA_H_KV       1
#define GQA_GROUP_SIZE 2   /* H_Q / H_KV */
#define GQA_S          64
#define GQA_D          64

void candidate_kernel(const uint8_t *Q, const int8_t *K, uint16_t *out);
#endif
