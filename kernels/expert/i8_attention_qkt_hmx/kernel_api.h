#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 attention QK^T scores on the HMX matrix engine, with the HMX requant epilogue.
 *   scores[i][j] = sum_d Q[i*D+d] * K[j*D+d]     (Q uint8, K int8, S=64, D=64; K is
 *                                                  stored [S,D] row-major -- row j IS
 *                                                  key vector j, so this computes
 *                                                  Q . K^T without an explicit transpose)
 *   out[i*S+j] = ((scores*17 + 8) >> 4) & 0xFFF   (HMX 0x40-config native requant:
 *                                                  scale 17/16, bias 0, 12-bit field)
 * Output is the uint16 12-bit field, bit-exact to the scalar reference.
 * S=64 is a 2x2 grid of 32x32 output tiles; each accumulates over 2 D-tiles (D=64=2*32).
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output tiles. Input ranges are chosen so
 * |requant result| < 2048 (the 12-bit field is exact). */
void candidate_kernel(const uint8_t *Q, const int8_t *K, uint16_t *out, int S, int D);
#endif
