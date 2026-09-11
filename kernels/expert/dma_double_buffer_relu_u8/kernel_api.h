#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Zero-point ReLU: out[i] = max(a[i] - 128, 0), computed as a uint8
 * saturating subtract (usat8(a[i]-128)), for i in [0, n). This is the
 * common "ReLU after a 128 zero-point quantization" pattern.
 * The achievability bar ping-pongs two VTCM buffers (input AND output) over
 * the N tiles: the next tile's input is DMA'd in, and the previous tile's
 * result is DMA'd out, while the current tile's ReLU is computed, hiding
 * DDR latency on BOTH directions behind compute. */
void candidate_kernel(const uint8_t *a, uint8_t *out, int n);
#endif
