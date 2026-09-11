#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 ReLU: out[i] = (x[i] > 0) ? x[i] : 0. Large-N / bandwidth-bound variant;
 * the achievability bar hides DDR latency with a rolling L2 prefetch (l2fetch). */
void candidate_kernel(const int8_t *x, int8_t *out, int n);
#endif
