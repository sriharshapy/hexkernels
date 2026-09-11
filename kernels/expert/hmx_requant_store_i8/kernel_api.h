#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* HMX int8 requant-store epilogue (decomposition micro-skill). Given the int32
 * accumulators produced by an HMX matmul, apply the fixed 0x40-config requant
 * (multiply by 17, round, arithmetic shift right 4) and SATURATE to signed int8:
 *   r   = (acc[i]*17 + 8) >> 4          (arithmetic shift; matches HMX 0x40 field)
 *   out[i] = clamp(r, -128, 127)         (signed int8 saturating store)
 * n is the element count (n == 1024). No matmul here -- this is the quantize+store
 * tail that turns int32 accumulators into an int8 tensor. Input range makes r
 * exceed the int8 range often, so the saturation must be correct.
 */
void candidate_kernel(const int32_t *acc, int8_t *out, int n);
#endif
