#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused 2x2 non-overlapping average pool (signed int8, truncate toward zero) then
 * requantize to int8. Window 2x2 / stride 2 are FIXED for this task.
 *   pool = (in[2oy][2ox] + in[2oy][2ox+1] + in[2oy+1][2ox] + in[2oy+1][2ox+1]) / 4
 *          (int32 accumulator, division truncates toward zero)
 *   v    = (int64_t)pool * mult
 *   half = shift > 0 ? (1LL << (shift-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *   r   += zp
 *   out  = saturate_to_int8(r)          // clamp to [-128,127]
 * mult, shift, zp are runtime parameters -- do NOT hardcode them.
 * Large-image / bandwidth-bound variant: the achievability bar DMA-tiles row blocks
 * of in[] into VTCM (double-buffered) and pools+requantizes the on-chip copies. */
void candidate_kernel(const int8_t *in, int8_t *out, int w, int h,
                      int32_t mult, int shift, int8_t zp);
#endif
