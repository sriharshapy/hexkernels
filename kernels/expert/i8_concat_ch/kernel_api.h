#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Channel concatenation: concat two tensors along the channel axis.
 * a: (C1, H, W) = (3, 8, 11), layout [C1][H][W].
 * b: (C2, H, W) = (5, 8, 11), layout [C2][H][W].
 * out: (C1+C2, H, W) = (8, 8, 11), layout [C1+C2][H][W].
 *
 * Index mapping:
 *   out[c * H*W + h*W + w] = a[c * H*W + h*W + w]          for c in [0, C1)
 *   out[c * H*W + h*W + w] = b[(c-C1) * H*W + h*W + w]     for c in [C1, C1+C2)
 *
 * Pure data movement, no arithmetic.
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out,
                      int C1, int C2, int H, int W);
#endif
