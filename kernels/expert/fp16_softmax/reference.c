/* PLAIN SCALAR fp16 row-wise softmax -- the DENOMINATOR baseline. No HVX
 * intrinsics, no vector types anywhere: scalar max-reduce, scalar exp,
 * scalar per-element divide by the row sum. */
#include "kernel_api.h"
#include <math.h>
#include <stddef.h>

void candidate_kernel(const hvx_hf *x, hvx_hf *out, int R, int C) {
    for (int r = 0; r < R; r++) {
        const hvx_hf *xr = x + (size_t)r * C;
        hvx_hf *outr = out + (size_t)r * C;

        float rowmax = (float)xr[0];
        for (int j = 1; j < C; j++) {
            float v = (float)xr[j];
            if (v > rowmax) rowmax = v;
        }

        float rowsum = 0.0f;
        float e[256];
        for (int j = 0; j < C; j++) {
            e[j] = expf((float)xr[j] - rowmax);
            rowsum += e[j];
        }

        for (int j = 0; j < C; j++)
            outr[j] = (hvx_hf)(e[j] / rowsum);
    }
}
