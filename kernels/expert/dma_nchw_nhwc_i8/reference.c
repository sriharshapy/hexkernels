/* Baseline: natural direct NCHW->NHWC scan. Loop order (n, c, h, w) makes
 * `in` sequential per channel (each channel is H*W contiguous bytes) but
 * `out` is written at stride C (scattered straight to DDR, no VTCM
 * staging). This is the speedup denominator: genuinely slow in the
 * memory-timing model because the writes are scattered -- exactly what the
 * banded DMA-chain + on-chip-transpose design (reused from
 * dma_2d_transpose_i8) fixes by moving the scatter on-chip into VTCM. Plain
 * scalar -- H*W=20000 is not 128-aligned per channel, so a per-channel HVX
 * vector load/store here would be misaligned; the natural competent
 * baseline for this scatter-write shape is a scalar byte loop (matching the
 * repo's other layout-conversion baselines, e.g. nchw_to_nhwc). */
#include <stdint.h>

void candidate_kernel(const int8_t *in, int8_t *out, int N, int C, int H, int W) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *src = in + ((long)n * C + c) * H * W;
            int8_t *dstbase = out + (long)n * H * W * C + c;  /* stride C per spatial idx */
            long hw = (long)H * W;
            for (long p = 0; p < hw; p++) dstbase[p * C] = src[p];
        }
    }
}
