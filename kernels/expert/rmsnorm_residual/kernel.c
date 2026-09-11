/* Fused RMSNorm-after-residual, int8, no floating point -- the HVX expert.
 *
 * Step 1, the saturating residual add, is the part that vectorises cleanly: it is
 * elementwise over int8 with exactly the clamp Q6_Vb_vadd_VbVb_sat performs in
 * hardware, so one vector op covers 128 lanes. n is 113, so a single vector covers the
 * whole array; the loop is written over ceil(n/128) anyway so the shape does not depend
 * on that. The 15 lanes past n are computed and then ignored -- every later pass reads
 * only the first n elements of the scratch, so the garbage never reaches the output.
 *
 * Steps 2-6 stay scalar, deliberately. The rms2 reduction is a horizontal sum over a
 * NON-multiple-of-128 length, and the normalise is a lookup into a 256-entry runtime
 * table followed by two round-half-up shifts that have to stay bit-exact -- the
 * 32-byte vlut32 gather cannot serve a 256-entry table without eight chained passes,
 * which is more work than the 113 scalar loads it would replace. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const uint8_t *inv_lut) {
    static int8_t t[256] HVX_ALIGN;   /* n rounded up to whole vectors */

    /* Step 1 [HVX]: t = saturating int8 add of x and residual */
    for (int v = 0; v < (n + 127) / 128; v++)
        ((HVX_Vector *)t)[v] = Q6_Vb_vadd_VbVb_sat(((const HVX_Vector *)x)[v],
                                                   ((const HVX_Vector *)residual)[v]);

    /* Step 2: rms2 = sum(t^2)/n over the REAL n elements only (no mean subtraction --
     * this is RMSNorm, not LayerNorm) */
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)t[i] * (int32_t)t[i];
    int32_t rms2 = s / n;
    int ridx = rms2 < 0 ? 0 : (rms2 > 255 ? 255 : rms2);
    int inv  = (int)inv_lut[ridx];

    /* Steps 4-6: per-element gamma scale, inverse-RMS scale, clamp */
    for (int i = 0; i < n; i++) {
        int32_t scaled = ((int32_t)t[i] * (int32_t)gamma[i] + 64) >> 7;
        int32_t normed = (scaled * inv + 128) >> 8;
        out[i] = (int8_t)(normed > 127 ? 127 : (normed < -128 ? -128 : normed));
    }
}
