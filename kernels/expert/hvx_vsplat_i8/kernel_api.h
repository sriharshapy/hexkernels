#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Broadcast a single int8 scalar to every element of an output vector,
 * n=1011 (tail path: 1011 = 7*128 + 115).
 * Semantics: out[i] = val for i in [0, n).
 * Matches Q6_Vb_vsplat_R (byte-granularity splat of the low 8 bits of a
 * 32-bit register into every byte lane of a vector).
 */
void candidate_kernel(int8_t val, int8_t *out, int n);
#endif
