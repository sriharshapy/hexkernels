#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Depth-to-space: inverse of space-to-depth. Rearrange channel dimension into spatial blocks.
 * Block factor b=2. Input (C_in,H_in,W_in)=(12,4,5). Output (C_in/b/b, H_in*b, W_in*b)=(3,8,10).
 *
 * Input channel c_in = c*b*b + bh*b + bw   (c in [0,C_out), bh,bw in [0,b))
 * Index mapping:
 *   in[c_in * H_in*W_in + oh*W_in + ow] = out[c*H_out*W_out + (oh*b+bh)*W_out + (ow*b+bw)]
 * Equivalently, the output receives:
 *   out[c*H_out*W_out + (oh*b+bh)*W_out + (ow*b+bw)] = in[c_in * H_in*W_in + oh*W_in + ow]
 *
 * Pure data movement, no arithmetic.
 */
void candidate_kernel(const int8_t *in, int8_t *out,
                      int C_in, int H_in, int W_in, int b);
#endif
