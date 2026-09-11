#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Row-strided depad copy (2D DMA): a[] is an H x rowstride buffer (each row
 * padded to `rowstride` bytes); out[] is the compact H x W buffer with the
 * padding removed:
 *   out[r*W + c] = a[r*rowstride + c]   for r in [0,H), c in [0,W), W <= rowstride
 *
 * This toolchain has no documented hardware "Type-1 2D descriptor" bit
 * layout, so the achievability bar builds the 2D/row-strided transfer out of
 * a CHAIN of per-row Type-0 uDMA descriptors (one descriptor per row, linked
 * via `next`, issued with a single Q6_dmstart_A call) that gathers a
 * row-group directly into a contiguous VTCM buffer, then DMAs that
 * contiguous buffer out to `out` in one flat transfer. Double-buffered
 * across row-groups. */
void candidate_kernel(const int8_t *a, int8_t *out, int H, int W, int rowstride);
#endif
