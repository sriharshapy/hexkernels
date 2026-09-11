#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * NCHW -> NHWC layout conversion (DMA/VTCM-staged), int8, N images:
 *   out[((n*H+h)*W+w)*C + c] = in[((n*C+c)*H+h)*W+w]
 *   for n in [0,N), c in [0,C), h in [0,H), w in [0,W)
 *
 * For a FIXED image n this is EXACTLY a C x (H*W) -> (H*W) x C matrix
 * transpose: channels are rows of H*W contiguous bytes in `in`; spatial
 * positions are rows of C contiguous bytes in `out`. This toolchain has no
 * hardware "2D descriptor" (only a flat Type-0 desc_t{next,ctrl,src,dst}),
 * so the achievability bar reuses the same banded gather-DMA-chain +
 * on-chip-transpose + flat-DMA-out design as dma_2d_transpose_i8, applied
 * per-image, with C playing the role of the small/chained dimension and
 * H*W playing the role of the large banded dimension.
 */
void candidate_kernel(const int8_t *in, int8_t *out, int N, int C, int H, int W);
#endif
