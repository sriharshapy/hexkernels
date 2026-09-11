#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused elementwise: out[i] = requant(relu(a[i] + b[i])) for i in [0, n).
 * a and b are int32 input arrays. Sum is computed in int32, then clamped to
 * [0, INT32_MAX] (ReLU), then requantized to int8 using round-half-away-from-zero:
 *   v    = (int64_t)sum * mult
 *   half = shift > 0 ? (1LL << (shift-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *   r   += zp
 *   out  = saturate_to_int8(r)   -- clamp to [-128, 127]
 * After ReLU the sum is >= 0 so v >= 0 when mult >= 0 (handle negative mult too).
 * mult, shift, zp are runtime params -- do NOT hardcode them. */
void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp);
#endif
