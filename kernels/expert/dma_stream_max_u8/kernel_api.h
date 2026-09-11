#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Streamed MAX reduction over a large uint8 array:
 *   out[0] = max(a[0], a[1], ..., a[n-1])     (unsigned uint8 max)
 *
 * n is large (working set exceeds L2), so this is a single DDR-bandwidth-
 * bound reduction pass. To go fast, DMA double-buffer a[] through VTCM in
 * tiles (prefetch tile c+1 while reducing tile c's max on-chip), hiding DDR
 * latency behind compute, then do a final small cross-lane reduction.
 */
void candidate_kernel(const uint8_t *a, int n, uint8_t *out);
#endif
