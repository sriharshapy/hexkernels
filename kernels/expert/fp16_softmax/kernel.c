/* HVX qf16 row-wise softmax -- the ACCELERATED expert (pure HVX, no matrix
 * engine). Per row (C=256 = 4 blocks of 64 lanes):
 *  1. max-reduce: running per-lane max across the 4 blocks
 *     (Q6_Vhf_vmax_VhfVhf, native at v68), then unpack the 64-lane result
 *     ONCE and finish the reduction in scalar (64 compares -- negligible).
 *  2. exp: HVX has no vector transcendental, so this is necessarily a
 *     scalar loop over the row (float conversion + expf), but it is fused
 *     with the running sum accumulation (no extra pass).
 *  3. normalize: broadcast 1/rowsum (bit-pattern splat) and multiply each
 *     of the 4 blocks in qf16 -- vectorized, replacing a per-element scalar
 *     divide with a per-block vector multiply. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

#define BLK 64   /* fp16 lanes per 128B HVX vector */

static inline HVX_Vector hf_splat_f(float f) {
    hvx_hf v = (hvx_hf)f;
    unsigned short bits = *(const unsigned short *)&v;
    return Q6_Vh_vsplat_R((int)bits);
}

void candidate_kernel(const hvx_hf *x, hvx_hf *out, int R, int C) {
    int nb = C / BLK;   /* C is a multiple of BLK (single-tile row, no remainder) */
    static hvx_hf ebuf[256] HVX_ALIGN;   /* scratch for exp(x-max), reused per row */

    for (int r = 0; r < R; r++) {
        const hvx_hf *xr = x + (size_t)r * C;
        const HVX_Vector *xv = (const HVX_Vector *)xr;

        /* 1. vectorized max-reduce across the nb blocks. */
        HVX_Vector maxv = xv[0];
        for (int b = 1; b < nb; b++) maxv = Q6_Vhf_vmax_VhfVhf(maxv, xv[b]);
        const hvx_hf *mp = (const hvx_hf *)&maxv;
        float rowmax = (float)mp[0];
        for (int j = 1; j < BLK; j++) if ((float)mp[j] > rowmax) rowmax = (float)mp[j];

        /* 2. scalar exp (unavoidable -- no HVX vector transcendental), fused
         * with the running sum. */
        float rowsum = 0.0f;
        for (int j = 0; j < C; j++) {
            float e = expf((float)xr[j] - rowmax);
            ebuf[j] = (hvx_hf)e;
            rowsum += e;
        }

        /* 3. vectorized normalize: multiply by the splatted 1/rowsum. */
        float inv_sum = 1.0f / rowsum;
        HVX_Vector invSumVec = hf_splat_f(inv_sum);
        const HVX_Vector *ev = (const HVX_Vector *)ebuf;
        HVX_Vector *ov = (HVX_Vector *)(out + (size_t)r * C);
        for (int b = 0; b < nb; b++) {
            HVX_Vector sc = Q6_Vqf16_vmpy_VhfVhf(ev[b], invSumVec);
            ov[b] = Q6_Vhf_equals_Vqf16(sc);
        }
    }
}
