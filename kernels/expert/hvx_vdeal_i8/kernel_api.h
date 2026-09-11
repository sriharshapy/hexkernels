#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-128-byte-block byte deal (de-interleave into even/odd halves),
 * n=768 (6 full 128-byte blocks; block-granular op, no scalar byte tail).
 * Semantics: for each block b (128 bytes at offset b*128), and for
 * i in [0, 64):
 *   out[b*128 + i]      = a[b*128 + 2*i]
 *   out[b*128 + 64 + i] = a[b*128 + 2*i + 1]
 * i.e. within each block, gather the even-indexed bytes into the low
 * half and the odd-indexed bytes into the high half (the exact inverse
 * of the vshuff interleave). Matches Q6_Vb_vdeal_Vb.
 */
void candidate_kernel(const int8_t *a, int8_t *out, int n);
#endif
