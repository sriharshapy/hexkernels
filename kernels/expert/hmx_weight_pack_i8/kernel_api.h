#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* HMX int8 WEIGHT crouton pack (a decomposition micro-skill of the full int8
 * HMX matmul). Given a row-major 32x32 weight tile B (int8, contraction index k
 * as row, output column j as col), write the 1024-byte 4-deep packed weight
 * buffer that the HMX `weight.b` mxmem load expects:
 *     out[ (k/4)*128 + j*4 + (k%4) ] = B[k*n + j]
 * i.e. 4 consecutive contraction elements are packed into 4 consecutive bytes,
 * 128 bytes per contraction-group x 8 groups = the dense 1024-byte tile. Every
 * output byte is written (no padding). n == 32; `out` is 1024 bytes. No HMX
 * instruction is required. (Helper hvx_hmx_i8_wgt_off(k,j) in harness_common.h.)
 */
void candidate_kernel(const int8_t *B, int8_t *out, int n);
#endif
