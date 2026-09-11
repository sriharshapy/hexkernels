#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-128-byte-block byte shuffle (interleave low/high halves), n=640
 * (5 full 128-byte blocks; block-granular op, no scalar byte tail --
 * see prompt.md for why).
 * Semantics: for each block b (128 bytes at offset b*128), and for
 * i in [0, 64):
 *   out[b*128 + 2*i]     = a[b*128 + i]
 *   out[b*128 + 2*i + 1] = a[b*128 + 64 + i]
 * i.e. within each block, interleave the low 64 bytes and high 64 bytes.
 * Matches Q6_Vb_vshuff_Vb (single-operand in-block byte shuffle).
 */
void candidate_kernel(const int8_t *a, int8_t *out, int n);
#endif
