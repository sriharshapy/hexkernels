#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Convert NHWC layout to NCHW layout (N=1, C=4, H=10, W=13).
 * in  [N,H,W,C]: in [h*W*C + w*C + c]
 * out [N,C,H,W]: out[c*H*W + h*W + w]
 * Pure data movement, no arithmetic.
 */
void candidate_kernel(const int8_t *in, int8_t *out, int C, int H, int W);
#endif
