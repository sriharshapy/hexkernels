#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * 2D zero-padding: pad a 2D array by P rows/cols on each side.
 * H=10, W=13, P=3 → output (H+2*P) x (W+2*P) = 16 x 19.
 *
 * For pixel (r, c) in the padded output:
 *   out[r * (W+2*P) + c] = in[(r-P) * W + (c-P)]  if P <= r < H+P and P <= c < W+P
 *   out[r * (W+2*P) + c] = 0                        otherwise (border)
 *
 * Pure data movement (zero-fill), no arithmetic on the values.
 */
void candidate_kernel(const int8_t *in, int8_t *out,
                      int H, int W, int P);
#endif
