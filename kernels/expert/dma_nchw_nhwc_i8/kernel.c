/* Solution 1 (= expert): per image, reuse the dma_2d_transpose_i8 banded
 * gather-DMA-chain design: C is small and fixed per call, so for each
 * TC-wide band of the flattened H*W spatial index, a CHAIN of C
 * per-channel Type-0 descriptors (linked via `next`) gathers the C x
 * band_width sub-rectangle into a contiguous VTCM buffer with a SINGLE
 * Q6_dmstart_A trigger. The band is transposed on-chip into a second VTCM
 * buffer, densely packed (row stride C) so the whole transposed band is
 * contiguous both in VTCM and in `out` -- a single flat descriptor DMAs it
 * out. The next band's input chain is prefetched (async) while the current
 * band is transposed + written out, double-buffered across bands, repeated
 * per image.
 *
 * TC=4096 (matching dma_2d_transpose_i8's retune): a first attempt at
 * TC=128 measured no meaningful speedup over baseline -- profiling showed
 * the bottleneck is NOT descriptor-chain overhead but the scalar
 * byte-by-byte on-chip transpose loop itself (each scalar VTCM element
 * access costs ~48 cycles in this sim's timing model, the same documented
 * pitfall as HMX crouton pack/unpack). Widening TC changes the descriptor
 * count but does NOT reduce the total scalar VTCM byte-ops (C*H*W
 * regardless of band size), so this is kept at the same widened band for
 * consistency with dma_2d_transpose_i8, not because it fixes the
 * bottleneck -- see that task's spec.json accelerable/mechanism_attribution
 * note. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define TC     4096  /* spatial-band width (flattened H*W positions per band) */
#define TCSLOT 4096  /* fixed VTCM row-slot stride for the gather chain */
#define MAXCHAIN 128 /* static chain capacity; C must be <= this */

static desc_t chain0[MAXCHAIN], chain1[MAXCHAIN];
static desc_t d_out;

static void build_chain(desc_t *chain, int C, const int8_t *img, long HW,
                         int p0, int tw, uint32_t vt_base) {
    for (int c = 0; c < C; c++) {
        chain[c].next = (c + 1 < C) ? (uint32_t)(uintptr_t)&chain[c + 1] : 0;
        chain[c].ctrl = (uint32_t)tw;
        chain[c].src  = (uint32_t)(uintptr_t)(img + (long)c * HW + p0);
        chain[c].dst  = vt_base + (uint32_t)(c * TCSLOT);
    }
}

static void transpose_image(const int8_t *img, int8_t *oimg, int C, long HW) {
    const uint32_t vt_in0 = VTCM_BASE;
    const uint32_t vt_in1 = VTCM_BASE + (uint32_t)(C * TCSLOT);
    const uint32_t vt_out = VTCM_BASE + (uint32_t)(2 * C * TCSLOT);

    int nbands = (int)((HW + TC - 1) / TC);
    if (nbands == 0) return;

    int tw0 = (TC <= HW) ? TC : (int)HW;
    build_chain(chain0, C, img, HW, 0, tw0, vt_in0);
    Q6_dmstart_A(&chain0[0]); Q6_R_dmwait();

    for (int g = 0; g < nbands; g++) {
        int p0 = g * TC;
        int tw = (p0 + TC <= HW) ? TC : (int)(HW - p0);
        uint32_t cur_in = (g & 1) ? vt_in1 : vt_in0;
        uint32_t nxt_in = (g & 1) ? vt_in0 : vt_in1;
        desc_t *nxt_chain = (g & 1) ? chain0 : chain1;

        if (g + 1 < nbands) {
            int p0n = (g + 1) * TC;
            int twn = (p0n + TC <= HW) ? TC : (int)(HW - p0n);
            build_chain(nxt_chain, C, img, HW, p0n, twn, nxt_in);
            Q6_dmstart_A(&nxt_chain[0]);
        }

        /* On-chip transpose: cur_in holds C rows x tw valid bytes (row
         * stride TCSLOT); write out[j][t] = in[t][j] densely (row stride C)
         * into vt_out. */
        {
            const int8_t *vin = (const int8_t *)(uintptr_t)cur_in;
            int8_t *vout = (int8_t *)(uintptr_t)vt_out;
            for (int j = 0; j < tw; j++)
                for (int t = 0; t < C; t++)
                    vout[j * C + t] = vin[t * TCSLOT + j];
        }

        if (g + 1 < nbands) Q6_R_dmwait();   /* finish the input prefetch */

        d_out.next = 0; d_out.ctrl = (uint32_t)(tw * C);
        d_out.src = vt_out; d_out.dst = (uint32_t)(uintptr_t)(oimg + (long)p0 * C);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }
}

void candidate_kernel(const int8_t *in, int8_t *out, int N, int C, int H, int W) {
    long HW = (long)H * W;
    for (int n = 0; n < N; n++)
        transpose_image(in + (long)n * C * HW, out + (long)n * HW * C, C, HW);
}
