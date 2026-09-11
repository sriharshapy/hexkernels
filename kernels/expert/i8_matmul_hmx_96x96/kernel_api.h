#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 96x96 matmul on the HMX matrix engine, with the HMX requant epilogue.
 *   acc[i][j] = sum_k A[i*N+k] * B[k*N+j]      (A uint8 0..3, B int8 -2..2, N=96)
 *   out[i*N+j] = ((acc*17 + 8) >> 4) & 0xFFF   (bias-config 0x40: scale 17/16,
 *                                               bias 0, 12-bit two's-comp field)
 * Output is the uint16 12-bit field, bit-exact to the scalar reference.
 * N=96 is a 3x3 grid of 32x32 output tiles; each output tile accumulates over
 * three K-tiles (K=96 = 3*32) with the proven 32x32 crouton packing issued in a
 * tiling loop -- the same recipe as the 64x64 sibling task, generalized to a
 * bigger tile grid so the crouton-load overhead amortizes over more compute.
 * (128x128/4x4 was authored and verified bit-exact but its O(n^3) scalar
 * reference blew the ~40s sim budget at ~58s; 96x96/3x3 is the largest grid
 * that stays in-budget while still exceeding the 64x64 sibling's amortization.)
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output tiles. Input ranges are chosen
 * so |requant result| < 2048 (the 12-bit field is exact, no saturation).
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, uint16_t *out, int n);
#endif
