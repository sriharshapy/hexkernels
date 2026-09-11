#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* HMX int8 ACTIVATION crouton pack (a decomposition micro-skill of the full
 * int8 HMX matmul). Given a row-major 32x32 activation tile A (uint8), write the
 * 2048-byte crouton-packed activation buffer `out` that the HMX `activation.ub`
 * mxmem load expects:
 *     out[ 2*((i/2)*64 + k*2 + (i&1)) + 1 ] = A[i*n + k]   (the int8 value lives
 *                                                            in the HIGH byte of
 *                                                            each fp16 slot)
 *     every other byte of out (the 1024 even/low bytes) = 0.
 * n == 32; `out` is a 2048-byte buffer. No HMX instruction is required -- this is
 * the pure index-remap the matmul does before the mxmem load. (Helper
 * hvx_hmx_i8_act_off(i,k) in harness_common.h gives the byte offset.)
 */
void candidate_kernel(const uint8_t *A, uint8_t *out, int n);
#endif
