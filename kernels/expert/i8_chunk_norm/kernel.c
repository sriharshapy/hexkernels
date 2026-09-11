/* Per-chunk (group) normalisation, int8 in and out, no floating point -- the HVX expert.
 *
 * The normalise step is the vectorisable part, as the prompt says: gamma and beta are
 * per-ELEMENT, so they load straight as vectors, and only mu and inv are per-chunk
 * scalars. The two statistics passes stay scalar -- mean and variance are horizontal
 * reductions, and inv is a 256-entry table lookup, past the 32-byte vlut32 gather.
 *
 * A chunk is n/G = 32 elements, narrower than a 128-byte vector, so the loop computes a
 * WHOLE vector's worth of lanes with the current chunk's mu and inv and then hands off
 * just that chunk's 32 bytes. Lanes belonging to other chunks are computed with the
 * wrong scalars and discarded -- at n = 128 that is four vector passes in total, and it
 * keeps the arithmetic uniform instead of building per-lane mu/inv vectors, which would
 * cost more scalar work than it saves.
 *
 * Lane staging, chosen so every intermediate sits in the narrowest lane that holds it:
 *
 *   d      = x - mu                    |d| <= 255                halfword
 *   scaled = (d*gamma + 64) >> 7       255*127 = 32385 < 32767   halfword (gamma is a
 *                                                                VECTOR, so this must
 *                                                                be a vector*vector
 *                                                                multiply -- word lanes
 *                                                                have no vmpyi_VwVw)
 *   normed = (scaled*inv + 128) >> 8   253*255 = 64515           WORD (inv is scalar)
 *   + beta, clamp to int8                                        halfword, then pack
 *
 * Q6_Vw_vmpyi_VwRh takes its multiplier from a scalar register as a halfword, and that
 * halfword must be REPLICATED INTO BOTH HALVES -- see RH(). Passing the bare value is
 * wrong on essentially every element rather than failing loudly; i8_hardswish_dma
 * builds its constant the same way. The Q6_Ww_vsxt_Vh / Q6_Vh_vasr_VwVwR_sat(hi, lo, 0)
 * round trip is an exact identity, so the word-lane detour reorders nothing.
 */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

/* a halfword multiplier for Q6_Vw_vmpyi_VwRh: replicated into both halves of Rt */
#define RH(v) (((int32_t)(v) & 0xFFFF) | (((int32_t)(v) & 0xFFFF) << 16))

/* one 64-lane halfword half: centre, gamma scale, inv scale, beta -- halfword lanes out */
static inline HVX_Vector affine_half(HVX_Vector h, HVX_Vector g, HVX_Vector b,
                                     HVX_Vector vmu, HVX_Vector v64h,
                                     HVX_Vector v128w, int32_t rinv) {
    HVX_Vector d  = Q6_Vh_vsub_VhVh(h, vmu);
    HVX_Vector sc = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(d, g), v64h), 7);
    HVX_VectorPair wp = Q6_Ww_vsxt_Vh(sc);
    HVX_Vector lo = Q6_Vw_vasr_VwR(
        Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(Q6_V_lo_W(wp), rinv), v128w), 8);
    HVX_Vector hi = Q6_Vw_vasr_VwR(
        Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(Q6_V_hi_W(wp), rinv), v128w), 8);
    return Q6_Vh_vadd_VhVh(Q6_Vh_vasr_VwVwR_sat(hi, lo, 0), b);
}

void candidate_kernel(const int8_t *x, int8_t *out, int n, int G,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    const int chunk = n / G;
    const HVX_Vector v64h  = Q6_Vh_vsplat_R(64);
    const HVX_Vector v128w = Q6_V_vsplat_R(128);
    static int8_t tmp[128] __attribute__((aligned(128)));

    for (int c = 0; c < G; c++) {
        const int8_t *xc = x     + c * chunk;
        int8_t       *oc = out   + c * chunk;

        /* Steps 1-4: mean, variance, LUT index, inverse -- scalar reductions */
        int32_t sum = 0;
        for (int j = 0; j < chunk; j++) sum += (int32_t)xc[j];
        int32_t mu = sum / chunk;

        int32_t vs = 0;
        for (int j = 0; j < chunk; j++) {
            int32_t d = (int32_t)xc[j] - mu;
            vs += d * d;
        }
        int32_t var = vs / chunk;
        int32_t vi  = (var < 0) ? 0 : (var > 255 ? 255 : var);
        const int32_t rinv = RH(inv_lut[vi]);

        /* Step 5 [HVX]: the per-element affine, over the vector holding this chunk */
        const int base = (c * chunk) & ~127;          /* vector that contains the chunk */
        const HVX_Vector vmu = Q6_Vh_vsplat_R(mu);
        HVX_VectorPair xp = Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(x     + base));
        HVX_VectorPair gp = Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(gamma + base));
        HVX_VectorPair bp = Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(beta  + base));
        HVX_Vector lo = affine_half(Q6_V_lo_W(xp), Q6_V_lo_W(gp), Q6_V_lo_W(bp),
                                    vmu, v64h, v128w, rinv);
        HVX_Vector hi = affine_half(Q6_V_hi_W(xp), Q6_V_hi_W(gp), Q6_V_hi_W(bp),
                                    vmu, v64h, v128w, rinv);
        *(HVX_Vector *)tmp = Q6_Vb_vpack_VhVh_sat(hi, lo);

        memcpy(oc, tmp + (c * chunk - base), chunk);
    }
}
