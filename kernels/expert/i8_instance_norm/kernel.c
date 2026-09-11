/* Instance normalisation, int8 in and out, no floating point -- the HVX expert.
 *
 * Everything the affine step needs is a per-CHANNEL scalar (mu, inv, gamma[c],
 * beta[c]) and each channel's spatial map is HW = 256 bytes = exactly two vectors, so
 * steps 5a-5d are a pure elementwise map and vectorise end to end. Only the two
 * statistics passes stay scalar: mean and variance are horizontal reductions, and the
 * inv lookup is into a 256-entry per-channel table, well past the 32-byte vlut32
 * gather.
 *
 * Two things here are easy to get wrong, and were, before the simulator said so:
 *
 * 1. Q6_Vw_vmpyi_VwRh takes its multiplier from a SCALAR register as a halfword, and
 *    that halfword must be REPLICATED INTO BOTH HALVES of the register -- see RH()
 *    below. Passing the bare value multiplies by a garbage operand and is wrong on
 *    essentially every element rather than failing loudly. i8_hardswish_dma builds its
 *    constant the same way, which is what gave this away.
 *
 * 2. The arithmetic runs entirely in WORD lanes, matching the reference's int32 chain
 *    step for step. Doing the first multiply-shift in halfword lanes looks safe on
 *    range grounds (255*127 = 32385 fits) but is a second place for a rounding or
 *    overflow difference to hide, and buys nothing at this size.
 *
 * The two VectorPair round trips used here were each checked to be an exact identity on
 * this data before being relied on: Q6_Wh_vunpack_Vb / Q6_Vb_vpack_VhVh_sat(hi, lo),
 * and Q6_Ww_vsxt_Vh / Q6_Vh_vasr_VwVwR_sat(hi, lo, 0). Because every coefficient is a
 * per-channel scalar the affine is elementwise, so the lo/hi convention of those pairs
 * cannot affect the result -- but only if the pack really is the inverse of the unpack,
 * and that is the part worth verifying rather than assuming.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

/* a halfword multiplier for Q6_Vw_vmpyi_VwRh: replicated into both halves of Rt */
#define RH(v) (((int32_t)(v) & 0xFFFF) | (((int32_t)(v) & 0xFFFF) << 16))

/* one word half: (d*gamma + 64) >> 7, then (that*inv + 128) >> 8, then beta */
static inline HVX_Vector norm_word(HVX_Vector dw, int32_t rgamma, int32_t rinv,
                                   HVX_Vector v64w, HVX_Vector v128w,
                                   HVX_Vector vbetaw) {
    HVX_Vector sc = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(dw, rgamma), v64w), 7);
    HVX_Vector nm = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(sc, rinv), v128w), 8);
    return Q6_Vw_vadd_VwVw(nm, vbetaw);
}

/* one 64-lane halfword half of the channel's map, returned in halfword lanes */
static inline HVX_Vector norm_half(HVX_Vector h, HVX_Vector vmu, int32_t rgamma,
                                   HVX_Vector v64w, HVX_Vector v128w, int32_t rinv,
                                   HVX_Vector vbetaw) {
    HVX_VectorPair wp = Q6_Ww_vsxt_Vh(Q6_Vh_vsub_VhVh(h, vmu));
    HVX_Vector lo = norm_word(Q6_V_lo_W(wp), rgamma, rinv, v64w, v128w, vbetaw);
    HVX_Vector hi = norm_word(Q6_V_hi_W(wp), rgamma, rinv, v64w, v128w, vbetaw);
    return Q6_Vh_vasr_VwVwR_sat(hi, lo, 0);
}

void candidate_kernel(const int8_t *x, int8_t *out,
                      int H, int W, int C,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    const int hw = H * W;
    const HVX_Vector v64w  = Q6_V_vsplat_R(64);
    const HVX_Vector v128w = Q6_V_vsplat_R(128);

    for (int c = 0; c < C; c++) {
        const int8_t  *xc   = x   + c * hw;
        int8_t        *oc   = out + c * hw;
        const uint8_t *clut = inv_lut + c * 256;

        /* Steps 1-2: mean then variance, both truncating, over the channel's map */
        int32_t sum = 0;
        for (int s = 0; s < hw; s++) sum += (int32_t)xc[s];
        int32_t mu = sum / hw;

        int32_t vs = 0;
        for (int s = 0; s < hw; s++) {
            int32_t d = (int32_t)xc[s] - mu;
            vs += d * d;
        }
        int32_t var = vs / hw;
        if (var < 0) var = 0;
        if (var > 255) var = 255;

        /* Steps 5a-5d [HVX]: centre, gamma scale, inverse-sigma scale, beta, clamp */
        const HVX_Vector vmu    = Q6_Vh_vsplat_R(mu);
        const HVX_Vector vbetaw = Q6_V_vsplat_R((int32_t)beta[c]);
        const int32_t    rgamma = RH(gamma[c]);
        const int32_t    rinv   = RH(clut[var]);

        for (int v = 0; v < hw / 128; v++) {
            HVX_VectorPair bp = Q6_Wh_vunpack_Vb(((const HVX_Vector *)xc)[v]);
            HVX_Vector lo = norm_half(Q6_V_lo_W(bp), vmu, rgamma, v64w, v128w, rinv, vbetaw);
            HVX_Vector hi = norm_half(Q6_V_hi_W(bp), vmu, rgamma, v64w, v128w, rinv, vbetaw);
            ((HVX_Vector *)oc)[v] = Q6_Vb_vpack_VhVh_sat(hi, lo);
        }
    }
}
