#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * im2col: lower a convolution into a matrix of receptive fields.
 *
 * Input:  in[C*H*W], layout [C][H][W], C=3, H=8, W=9.
 * Kernel: K=3 (square), stride S=1, pad P=0.
 * Output height: OH = (H-K)/S+1 = 6
 * Output width:  OW = (W-K)/S+1 = 7
 * Output matrix out[C*K*K][OH*OW] = out[27][42], stored ROW-MAJOR.
 *
 * Index mapping:
 *   row = c*K*K + kh*K + kw    (channel c, kernel row kh, kernel col kw)
 *   col = oh*OW + ow           (output spatial position)
 *   out[row * OH*OW + col] = in[c*H*W + (oh*S + kh)*W + (ow*S + kw)]
 *
 * Pure data movement, no arithmetic.
 */
void candidate_kernel(const int8_t *in, int8_t *out,
                      int C, int H, int W,
                      int K, int S,
                      int OH, int OW);
#endif
