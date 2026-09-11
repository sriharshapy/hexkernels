#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Saturating add-constant, DMA round-trip through VTCM:
 * out[i] = min(a[i] + 50, 255)  (uint8 saturating add) for i in [0, n).
 * The achievability bar issues a uDMA transfer and explicitly *polls* for
 * completion (Q6_R_dmpoll) in a spin loop instead of blocking on
 * Q6_R_dmwait, both for the DMA-in (DDR->VTCM) and the DMA-out (VTCM->DDR)
 * of each tile. Double-buffered across tiles. */
void candidate_kernel(const uint8_t *a, uint8_t *out, int n);
#endif
