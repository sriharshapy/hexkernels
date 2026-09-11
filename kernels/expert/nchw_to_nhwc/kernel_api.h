#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Convert NCHW layout to NHWC layout (N=1, C=4, H=10, W=13).
 * in  [N,C,H,W]: in [c*H*W + h*W + w]
 * out [N,H,W,C]: out[h*W*C + w*C + c]
 * Pure data movement, no arithmetic.
 */
void candidate_kernel(const int8_t *in, int8_t *out, int C, int H, int W);
#endif
