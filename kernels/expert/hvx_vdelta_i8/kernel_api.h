#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-128-byte-block byte reversal via the vdelta permutation network,
 * n=896 (7 full 128-byte blocks; block-granular op, no scalar byte tail).
 * Semantics: for each block b (128 bytes at offset b*128), and for
 * i in [0, 128):
 *   out[b*128 + i] = a[b*128 + 127 - i]
 * i.e. reverse the byte order within each 128-byte block.
 * Matches Q6_V_vdelta_VV(Vu, Vctrl) with Vctrl = every byte lane 0xFF
 * (the all-ones control vector drives every stage of the delta swap
 * network to swap, which for HVX's vdelta yields a full 128-byte
 * reversal). Use Q6_Vb_vsplat_R(0xFF) to build Vctrl.
 */
void candidate_kernel(const int8_t *a, int8_t *out, int n);
#endif
