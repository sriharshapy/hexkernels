/*
 * HVX channel-mixing block (MLP-Mixer style): int8 expand-GELU-project.
 *
 * Dims: TOKENS=8, CHANNELS=16, D_FF=16  (all small, K=16 in both GEMMs)
 *
 * Strategy:
 *   Both GEMMs are 16×16 int8 with K=16 (4 k-groups of 4).
 *   Pretranspose W1 and W2 into vrmpy-friendly blocks:
 *     For k-group kg, output neuron j in [0,15]:
 *       Wt1[kg*64 + j*4 + q] = W1[j*CHANNELS + kg*4+q]   (q=0..3)
 *       Wt2[kg*64 + j*4 + q] = W2[j*D_FF    + kg*4+q]
 *   Each k-group block is 16*4=64 bytes; all 4 groups = 256 bytes.
 *   Because N=16, each block occupies 64 bytes = half a vector lane range:
 *   load it into lo half of a 128-byte aligned zero-padded buffer.
 *
 *   For each token and each GEMM:
 *     - Load input row (16 bytes), splat each 4-byte chunk as int32 across vector.
 *     - Q6_Vw_vrmpyacc_VwVbVb accumulates 16 word lanes.
 *     - After K reduction, result is in the lo 16 word-lanes of one HVX_Vector.
 *     - Scalar requant + LUT (tiny, 16 elements per token) in scalar cleanup.
 *
 *   Requant is scalar int64 to match reference exactly.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

#define TOKENS   8
#define CHANNELS 16
#define D_FF     16
#define KGROUPS  4   /* K=16 / 4 */

static int8_t __attribute__((aligned(128))) Wt1[KGROUPS * 128]; /* padded to 128B per kgroup */
static int8_t __attribute__((aligned(128))) Wt2[KGROUPS * 128];

static inline int8_t requant_scalar(int32_t acc, int32_t bias_v, int32_t mult, int shift) {
    int64_t biased = (int64_t)acc + (int64_t)bias_v;
    int64_t v      = biased * (int64_t)mult;
    int64_t half   = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q      = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

void candidate_kernel(const int8_t  *X,
                      const int8_t  *W1,
                      const int32_t *b1,
                      const int8_t  *W2,
                      const int32_t *b2,
                      const int8_t  *gelu_lut,
                      int8_t        *out,
                      int32_t mult1, int shift1,
                      int32_t mult2, int shift2)
{
    /*
     * Pretranspose W1 and W2.
     *
     * W1[m*CHANNELS + c] → Wt1 layout:
     *   For k-group kg (kg=0..3), output m in [0..15]:
     *     Wt1[kg*128 + m*4 + q] = W1[m*CHANNELS + kg*4 + q]   (q=0..3)
     *   Padding: m=16..31 in each 128B block → zeroed.
     *
     * Same for W2[c*D_FF + m].
     */
    memset(Wt1, 0, sizeof(Wt1));
    memset(Wt2, 0, sizeof(Wt2));

    for (int kg = 0; kg < KGROUPS; kg++) {
        int8_t *blk1 = Wt1 + kg * 128;
        int8_t *blk2 = Wt2 + kg * 128;
        for (int m = 0; m < D_FF; m++) {
            blk1[m*4+0] = W1[m*CHANNELS + kg*4+0];
            blk1[m*4+1] = W1[m*CHANNELS + kg*4+1];
            blk1[m*4+2] = W1[m*CHANNELS + kg*4+2];
            blk1[m*4+3] = W1[m*CHANNELS + kg*4+3];
        }
        for (int c = 0; c < CHANNELS; c++) {
            blk2[c*4+0] = W2[c*D_FF + kg*4+0];
            blk2[c*4+1] = W2[c*D_FF + kg*4+1];
            blk2[c*4+2] = W2[c*D_FF + kg*4+2];
            blk2[c*4+3] = W2[c*D_FF + kg*4+3];
        }
    }

    /* Accumulation buffers: 32 int32 slots (HVX vector = 32 words), use lo 16 */
    int32_t accbuf1[32] __attribute__((aligned(128)));
    int32_t accbuf2[32] __attribute__((aligned(128)));

    int8_t y1[D_FF] __attribute__((aligned(16)));

    for (int t = 0; t < TOKENS; t++) {
        const int8_t *xrow = X + t * CHANNELS;

        /* ---- Step 1: expand GEMM (D_FF outputs, CHANNELS=16 inputs) ---- */
        HVX_Vector vacc1 = Q6_V_vzero();

        for (int kg = 0; kg < KGROUPS; kg++) {
            const int k0 = kg * 4;
            /* Pack xrow[k0..k0+3] into int32 word (byte = input, splat across all lanes) */
            int32_t a4 = (int32_t)((uint8_t)xrow[k0+0])
                       | ((int32_t)((uint8_t)xrow[k0+1]) <<  8)
                       | ((int32_t)((uint8_t)xrow[k0+2]) << 16)
                       | ((int32_t)((uint8_t)xrow[k0+3]) << 24);
            HVX_Vector va = Q6_V_vsplat_R(a4);
            HVX_Vector vb = *(const HVX_Vector *)(Wt1 + kg * 128);
            vacc1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1, va, vb);
        }

        /* Store accumulator for scalar requant + LUT */
        *(HVX_Vector *)accbuf1 = vacc1;

        for (int m = 0; m < D_FF; m++) {
            int8_t sat1 = requant_scalar(accbuf1[m], b1[m], mult1, shift1);
            uint8_t idx = (uint8_t)((int)sat1 + 128);
            y1[m] = gelu_lut[idx];
        }

        /* ---- Step 2: project GEMM (CHANNELS outputs, D_FF=16 inputs) ---- */
        HVX_Vector vacc2 = Q6_V_vzero();

        for (int kg = 0; kg < KGROUPS; kg++) {
            const int k0 = kg * 4;
            int32_t a4 = (int32_t)((uint8_t)y1[k0+0])
                       | ((int32_t)((uint8_t)y1[k0+1]) <<  8)
                       | ((int32_t)((uint8_t)y1[k0+2]) << 16)
                       | ((int32_t)((uint8_t)y1[k0+3]) << 24);
            HVX_Vector va = Q6_V_vsplat_R(a4);
            HVX_Vector vb = *(const HVX_Vector *)(Wt2 + kg * 128);
            vacc2 = Q6_Vw_vrmpyacc_VwVbVb(vacc2, va, vb);
        }

        *(HVX_Vector *)accbuf2 = vacc2;

        int8_t *out_row = out + t * CHANNELS;
        for (int c = 0; c < CHANNELS; c++) {
            out_row[c] = requant_scalar(accbuf2[c], b2[c], mult2, shift2);
        }
    }
}
