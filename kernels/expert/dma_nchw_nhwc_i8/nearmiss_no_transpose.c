/* Near-miss: correct banded gather-DMA-chain staging (input chain lands,
 * VTCM addressing, output DMA sizes all right), but the on-chip "transpose"
 * step does NOT swap the channel/spatial index -- it writes vout[t*TC + j]
 * instead of vout[j*C + t], i.e. it copies the band into a
 * differently-shaped VTCM region without actually transposing it. The
 * output DMA then reads that buffer as if it held the (band_width x C)
 * NHWC layout, so the emitted bytes are garbled/wrong (a row-major C x TC
 * copy read back as TC x C). Compiles fine; fails bit-exact. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define TC     128
#define TCSLOT 128
#define MAXCHAIN 128

static desc_t chain[MAXCHAIN];
static desc_t d_out;

static void transpose_image(const int8_t *img, int8_t *oimg, int C, long HW) {
    const uint32_t vt_in  = VTCM_BASE;
    const uint32_t vt_out = VTCM_BASE + (uint32_t)(C * TCSLOT);

    int nbands = (int)((HW + TC - 1) / TC);

    for (int g = 0; g < nbands; g++) {
        int p0 = g * TC;
        int tw = (p0 + TC <= HW) ? TC : (int)(HW - p0);

        for (int c = 0; c < C; c++) {
            chain[c].next = (c + 1 < C) ? (uint32_t)(uintptr_t)&chain[c + 1] : 0;
            chain[c].ctrl = (uint32_t)tw;
            chain[c].src  = (uint32_t)(uintptr_t)(img + (long)c * HW + p0);
            chain[c].dst  = vt_in + (uint32_t)(c * TCSLOT);
        }
        Q6_dmstart_A(&chain[0]); Q6_R_dmwait();

        {
            const int8_t *vin = (const int8_t *)(uintptr_t)vt_in;
            int8_t *vout = (int8_t *)(uintptr_t)vt_out;
            /* BUG: forgot to transpose -- just re-copies without swapping
             * the channel/spatial index. */
            for (int t = 0; t < C; t++)
                for (int j = 0; j < tw; j++)
                    vout[t * TC + j] = vin[t * TCSLOT + j];
        }

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
