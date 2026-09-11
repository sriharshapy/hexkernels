/*
 * HVX RNN cell (int8) — Elman: h_t[j] = tanh_lut[requant(Wx[j,:]*x + Wh[j,:]*h_prev + b[j])]
 *
 * Strategy (H=32, I=32):
 *   - All 32 int32 accumulators fit in ONE HVX vector (32 word lanes × 4 bytes = 128B).
 *   - Pretranspose Wx and Wh into 128B-aligned blocks of 4 rows × 32 cols.
 *     Block layout: for k-group kg, output row j, sub-index q:
 *       Wxt[kg*128 + j*4 + q] = Wx[(kg*4+q)*I + j]  (transposed: rows→cols)
 *     Wait — we need the COLUMN dimension to be H=32 so that vrmpy gives us acc[j].
 *     Actually the matvec is: for each j in [0,H):
 *       acc[j] += sum_{k} Wx[j*I+k] * x[k]
 *     We want ALL j simultaneously. For k-group of 4 (k0..k0+3):
 *       Contribution to acc[j] += Wx[j*I+k0]*x[k0] + Wx[j*I+k0+1]*x[k0+1] + ...
 *     This is a dot(Wx_col_block[:,k0:k0+4], x[k0:k0+4]) per output j.
 *     Pack: Wt_block[j*4+q] = Wx[j*I+(k0+q)] for j=0..31, q=0..3.
 *     Splat: splat(x[k0..k0+3] packed as int32 word) across all 32 word lanes.
 *     vrmpy(acc, splat_x, Wt_block): each word lane j += sum_q(x_byte_q * Wt_byte_q)
 *       but vrmpy does signed×unsigned byte pairs → Q6_Vw_vrmpyacc_VwVbVb
 *       interprets second as unsigned. We have both signed — use the signed version.
 *     Use Q6_Vw_vrmpyacc_VwVbVb(acc, vB, vA) where vB is the weight (signed) and
 *     vA is the splat of input (signed is OK here since vrmpy uses signed×signed if
 *     we splat in the B slot). Actually: Q6_Vw_vrmpyacc_VwVbVb treats BOTH as signed int8.
 *   - After Wx+Wh matvecs: add bias vector (loaded as HVX_Vector).
 *   - Scalar requant + tanh_lut for 32 outputs.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(
    const int8_t  *Wx,
    const int8_t  *Wh,
    const int32_t *b,
    const int8_t  *x,
    const int8_t  *h_prev,
    int8_t        *h_t,
    int            H,
    int            I,
    int32_t        mult,
    int            shift,
    int8_t         zp,
    const int8_t  *tanh_lut
) {
    /*
     * For H=32, I=32:
     *   Wx is [32×32], Wh is [32×32].
     *   We process K=I=32 in 8 groups of 4 for Wx,
     *              K=H=32 in 8 groups of 4 for Wh.
     *
     * Transposed layout for Wxt:
     *   For k-group kg (0..7), output row j (0..31), sub q (0..3):
     *     Wxt[kg*128 + j*4 + q] = Wx[j*I + (kg*4+q)]
     *   => one 128B HVX_Vector per k-group.
     *
     * Transposed layout for Wht:
     *   Wht[kg*128 + j*4 + q] = Wh[j*H + (kg*4+q)]
     */
    static int8_t __attribute__((aligned(128))) Wxt[8 * 128]; /* 8 groups × 128B */
    static int8_t __attribute__((aligned(128))) Wht[8 * 128];

    int num_kgroups_x = I >> 2;  /* I/4 groups */
    int num_kgroups_h = H >> 2;  /* H/4 groups */

    /* Build transposed Wx */
    for (int kg = 0; kg < num_kgroups_x; kg++) {
        int8_t *blk = Wxt + kg * 128;
        int k0 = kg * 4;
        for (int j = 0; j < H; j++) {
            blk[j*4+0] = Wx[j*I + k0+0];
            blk[j*4+1] = Wx[j*I + k0+1];
            blk[j*4+2] = Wx[j*I + k0+2];
            blk[j*4+3] = Wx[j*I + k0+3];
        }
    }

    /* Build transposed Wh */
    for (int kg = 0; kg < num_kgroups_h; kg++) {
        int8_t *blk = Wht + kg * 128;
        int k0 = kg * 4;
        for (int j = 0; j < H; j++) {
            blk[j*4+0] = Wh[j*H + k0+0];
            blk[j*4+1] = Wh[j*H + k0+1];
            blk[j*4+2] = Wh[j*H + k0+2];
            blk[j*4+3] = Wh[j*H + k0+3];
        }
    }

    /* Load bias as HVX_Vector (H=32 int32s = 128B) */
    HVX_Vector vacc = *(const HVX_Vector *)b;

    /* Wx matvec: accumulate Wx[j,:]*x for all j simultaneously */
    for (int kg = 0; kg < num_kgroups_x; kg++) {
        int k0 = kg * 4;
        /* Splat x[k0..k0+3] packed as int32 word across all 32 word lanes */
        int32_t xword = (int32_t)((uint8_t)x[k0+0])
                      | (int32_t)((uint8_t)x[k0+1] << 8)
                      | (int32_t)((uint8_t)x[k0+2] << 16)
                      | (int32_t)((uint8_t)x[k0+3] << 24);
        HVX_Vector vx = Q6_V_vsplat_R(xword);
        HVX_Vector vw = *(const HVX_Vector *)(Wxt + kg * 128);
        /* Q6_Vw_vrmpyacc_VwVbVb: acc[j] += sum_q( vw_byte[j*4+q] * vx_byte[j*4+q] )
         * both treated as signed int8 */
        vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, vw, vx);
    }

    /* Wh matvec: accumulate Wh[j,:]*h_prev for all j simultaneously */
    for (int kg = 0; kg < num_kgroups_h; kg++) {
        int k0 = kg * 4;
        int32_t hword = (int32_t)((uint8_t)h_prev[k0+0])
                      | (int32_t)((uint8_t)h_prev[k0+1] << 8)
                      | (int32_t)((uint8_t)h_prev[k0+2] << 16)
                      | (int32_t)((uint8_t)h_prev[k0+3] << 24);
        HVX_Vector vh = Q6_V_vsplat_R(hword);
        HVX_Vector vw = *(const HVX_Vector *)(Wht + kg * 128);
        vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, vw, vh);
    }

    /* Extract 32 int32 accumulators and do scalar requant + LUT */
    int32_t accbuf[32] __attribute__((aligned(128)));
    *(HVX_Vector *)accbuf = vacc;

    int64_t half_val = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int32_t izp = (int32_t)zp;

    for (int j = 0; j < H; j++) {
        int32_t rv = accbuf[j];
        int64_t v  = (int64_t)rv * (int64_t)mult;
        int64_t r;
        if (v >= 0) r = (v + half_val) >> shift;
        else        r = -((-v + half_val) >> shift);
        r += izp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        h_t[j] = tanh_lut[(uint8_t)(int8_t)r];
    }
}
