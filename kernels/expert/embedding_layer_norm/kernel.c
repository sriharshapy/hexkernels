/*
 * embedding_layer_norm HVX candidate kernel — v9
 *
 * T=32 tokens, D=64 embedding dim, VOCAB=256.
 * Processes 2 tokens per outer iteration: the pair (i, i+1) fills one 128B output
 * vector exactly (2×64 bytes), enabling a single aligned vmem store.
 *
 * HVX operations used for real compute:
 *  - Q6_Vw_vrmpyacc_VwVbVb: 4-wide int8 dot-product for sum reduction (→ mu)
 *  - Q6_Ww_vmpy_VhVh: int16×int16→int32 pair for d^2 sum (→ var)
 *  - Q6_Vh_vmpyi_VhVh: int16×int16→int16 low for d*gamma (fits int16)
 *  - Q6_Vw_vmpyi_VwRh: int32×scalar-int16 for scaled*inv
 *  - Q6_Vh_vpack_VwVw_sat: int32→int16 pack with sat
 *  - Q6_Vb_vpack_VhVh_sat: int16→int8 pack with sat (final output)
 */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>

/* Horizontal sum of all 32 int32 lanes (used from i16_dot_i32 sol_01) */
static inline int32_t hsum_Vw(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

/* Process one token: return output as int16 vector (D=64 lanes 0..63 are valid) */
static inline HVX_Vector process_token(
        const int8_t *row,
        HVX_Vector vgamma_h,    /* int16 gamma[0..63] */
        HVX_Vector vbeta_h,     /* int16 beta[0..63] */
        const uint8_t *inv_lut,
        HVX_Vector vones,       /* all-ones int8 */
        HVX_Vector vzero,       /* all-zeros */
        HVX_VectorPred qlo64    /* lower 64 bytes predicate */
) {
    int D = 64;

    /* Load 64-byte row (aligned to 64B, possibly at offset 0 or 64 within 128B block) */
    uintptr_t rblk = (uintptr_t)row & ~(uintptr_t)127;
    int roff = (int)((uintptr_t)row - rblk);
    HVX_Vector vblk = *(const HVX_Vector *)rblk;
    HVX_Vector vrow;
    if (roff == 0) {
        vrow = vblk;
    } else {
        vrow = Q6_V_vlalign_VVR(vblk, vblk, 64);
    }
    /* Zero upper 64 bytes so reductions don't include garbage */
    vrow = Q6_V_vmux_QVV(qlo64, vrow, vzero);

    /* ---- Sum reduction → mu ---- */
    HVX_Vector vsum_w = Q6_Vw_vrmpyacc_VwVbVb(vzero, vrow, vones);
    int32_t total_sum = hsum_Vw(vsum_w);
    int32_t mu = total_sum / D;

    /* ---- d[j] = row[j] - mu (int16) ---- */
    HVX_Vector vrow_h = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vrow));  /* int16 row[0..63] */
    HVX_Vector vmu_h = Q6_Vh_vsplat_R((int16_t)mu);
    HVX_Vector vd_h = Q6_Vh_vsub_VhVh(vrow_h, vmu_h);

    /* ---- Variance = sum(d^2) / D ---- */
    HVX_VectorPair vsq_pair = Q6_Ww_vmpy_VhVh(vd_h, vd_h);
    /* lo = even-indexed d^2, hi = odd-indexed d^2 (interleaved) */
    /* Sum lo+hi gives sum of all d^2 values (order irrelevant for reduction) */
    HVX_Vector vsq_sum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(vsq_pair), Q6_V_hi_W(vsq_pair));
    int32_t var_sum = hsum_Vw(vsq_sum);
    int32_t var = var_sum / D;

    int32_t v_idx = (var < 0) ? 0 : (var > 255) ? 255 : var;
    uint8_t inv = inv_lut[v_idx];

    /* ---- scaled[j] = (d[j]*gamma[j]+64)>>7 (int16) ---- */
    /* d[j] in [-255,255], gamma[j] in [-128,127]; product in [-32640,32512] → fits int16 */
    HVX_Vector vdg_h = Q6_Vh_vmpyi_VhVh(vd_h, vgamma_h);
    vdg_h = Q6_Vh_vadd_VhVh(vdg_h, Q6_Vh_vsplat_R(64));
    HVX_Vector vscaled_h = Q6_Vh_vasr_VhR(vdg_h, 7);

    /* ---- normed[j] = (scaled[j]*inv+128)>>8 (via int32) ---- */
    /* scaled may overflow int16 when multiplied by inv (0..255), use int32 */
    HVX_VectorPair vscaled_wp = Q6_Ww_vunpack_Vh(vscaled_h);
    HVX_Vector vsc_lo = Q6_V_lo_W(vscaled_wp);  /* int32 scaled[0..31] */
    HVX_Vector vsc_hi = Q6_V_hi_W(vscaled_wp);  /* int32 scaled[32..63] */

    /* Broadcast inv as scalar int16 in both halves of 32-bit reg */
    int32_t inv32 = (int32_t)(uint16_t)inv | ((int32_t)(uint16_t)inv << 16);
    HVX_Vector vsn_lo = Q6_Vw_vmpyi_VwRh(vsc_lo, inv32);
    HVX_Vector vsn_hi = Q6_Vw_vmpyi_VwRh(vsc_hi, inv32);

    HVX_Vector v128w = Q6_V_vsplat_R(128);
    vsn_lo = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(vsn_lo, v128w), 8);
    vsn_hi = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(vsn_hi, v128w), 8);

    /* Pack int32→int16 with saturation (sequential: [0..31] from lo, [32..63] from hi) */
    HVX_Vector vnormed_h = Q6_Vh_vpack_VwVw_sat(vsn_hi, vsn_lo);

    /* ---- out[j] = clamp(normed[j]+beta[j], -128, 127) in int16 ---- */
    return Q6_Vh_vadd_VhVh(vnormed_h, vbeta_h);
}

void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int D,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    /* Preload gamma and beta as int16 vectors */
    HVX_Vector vgamma_h = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*(const HVX_Vector *)gamma));
    HVX_Vector vbeta_h  = Q6_V_lo_W(Q6_Wh_vunpack_Vb(*(const HVX_Vector *)beta));

    HVX_Vector vones  = Q6_Vb_vsplat_R(1);
    HVX_Vector vzero  = Q6_V_vzero();
    HVX_VectorPred qlo64 = Q6_Q_vsetq2_R(64);

    /* Process 2 tokens per iteration → one aligned 128B store */
    for (int i = 0; i < T; i += 2) {
        const int8_t *row0 = table + (int)idx[i]   * D;
        const int8_t *row1 = table + (int)idx[i+1] * D;

        /* Process each token → int16 result vector (valid in lanes 0..63) */
        HVX_Vector vout0_h = process_token(row0, vgamma_h, vbeta_h, inv_lut,
                                            vones, vzero, qlo64);
        HVX_Vector vout1_h = process_token(row1, vgamma_h, vbeta_h, inv_lut,
                                            vones, vzero, qlo64);

        /* Pack both int16 results to int8 with saturation:
         * Q6_Vb_vpack_VhVh_sat(Vu, Vv):
         *   result[0..63]   = sat8(Vv[0..63])  = sat8(vout0_h[0..63])
         *   result[64..127] = sat8(Vu[0..63])  = sat8(vout1_h[0..63])
         * So: result = [out[i], out[i+1]] as a 128B vector (2 tokens concatenated).
         */
        HVX_Vector vout_b = Q6_Vb_vpack_VhVh_sat(vout1_h, vout0_h);

        /* Store 128B aligned (out + i*D = out + i*64 is 128B-aligned for even i) */
        *(HVX_Vector *)(out + i * D) = vout_b;
    }
}
