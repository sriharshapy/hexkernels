/* Near-miss: identical HVX max-reduce + scalar exp as expert.c, but DROPS
 * the final normalize-by-rowsum step (plausible bug: forgetting the
 * denominator, common when refactoring a softmax into "exp" + "normalize"
 * passes). Compiles, genuinely uses HVX (the max-reduce), but the output
 * doesn't sum to 1 per row -> must FAIL the tolerance gate. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

#define BLK 64

void candidate_kernel(const hvx_hf *x, hvx_hf *out, int R, int C) {
    int nb = C / BLK;

    for (int r = 0; r < R; r++) {
        const hvx_hf *xr = x + (size_t)r * C;
        const HVX_Vector *xv = (const HVX_Vector *)xr;

        HVX_Vector maxv = xv[0];
        for (int b = 1; b < nb; b++) maxv = Q6_Vhf_vmax_VhfVhf(maxv, xv[b]);
        const hvx_hf *mp = (const hvx_hf *)&maxv;
        float rowmax = (float)mp[0];
        for (int j = 1; j < BLK; j++) if ((float)mp[j] > rowmax) rowmax = (float)mp[j];

        hvx_hf *outr = out + (size_t)r * C;
        for (int j = 0; j < C; j++) {
            float e = expf((float)xr[j] - rowmax);
            outr[j] = (hvx_hf)e;   /* bug: never divides by rowsum */
        }
    }
}
