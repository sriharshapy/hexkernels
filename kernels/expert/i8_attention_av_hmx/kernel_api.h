#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 attention A.V (scores x values) on the HMX matrix engine, with the HMX
 * requant epilogue.
 *   out[i][j] = sum_k P[i*S+k] * V[k*D+j]      (P uint8 0..7 -- post-softmax
 *                                                attention-weight proxy, V
 *                                                int8 -3..3, S=64, D=64)
 *   out[i][j] = ((acc*17 + 8) >> 4) & 0xFFF     (0x40-config HMX requant field)
 * Output is the uint16 12-bit field, bit-exact to the scalar reference.
 * S=64,D=64 is a 2x2 grid of 32x32 output tiles; each output tile accumulates
 * over two S-tiles (S=64 = 2*32, the reduction/contraction axis over keys).
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output tiles. Input ranges are
 * chosen so |requant result| < 2048 (the 12-bit field is exact). */
void candidate_kernel(const uint8_t *P, const int8_t *V, uint16_t *out, int S, int D);
#endif
