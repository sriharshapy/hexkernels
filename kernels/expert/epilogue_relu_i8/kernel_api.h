#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 ReLU epilogue over a small IN-CACHE tile (no DMA/l2fetch):
 *   out[i] = (x[i] > 0) ? x[i] : 0
 * n=1000 (NOT a multiple of 128 -- real tail path). Distinct from
 * i8_relu_l2fetch (that task is the large-N, DDR-bandwidth-bound streaming
 * variant that hides latency with a rolling l2fetch prefetch); this task
 * models a small tile-epilogue context (e.g. the ReLU applied right after a
 * conv/matmul tile is already resident in L1/L2) -- pure HVX compute, no
 * memory-hiding mechanism needed or expected.
 */
void candidate_kernel(const int8_t *x, int8_t *out, int n);
#endif
