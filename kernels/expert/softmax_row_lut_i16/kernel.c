#include "kernel_api.h"
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

/* s1: HVX-vectorized index derivation using NATIVE 16-bit saturating
 * subtract (Q6_Vh_vsub_VhVh_sat) -- no widening needed since saturation to
 * INT16_MIN on extreme underflow is still correctly "< -255" once clamped.
 * Steps 3-5 (LUT gather / sum / normalize) are scalar: exp_lut has uint16
 * entries, so the byte-oriented vlut32 gather idiom (which produces one
 * gathered BYTE per index) does not directly apply to a 16-bit table, and
 * C=113 is small enough that the scalar gather+reduce is cheap. */
void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                      const uint16_t *exp_lut) {
    int16_t xrow_buf[128] __attribute__((aligned(128)));
    uint8_t idx_buf[128]  __attribute__((aligned(128)));
    uint16_t evals[128];

    for (int r = 0; r < R; r++) {
        const int16_t *row  = x   + (long)r * C;
        int16_t       *orow = out + (long)r * C;

        /* Step 1: scalar row max (safe reduction). */
        int16_t m = row[0];
        for (int j = 1; j < C; j++) if (row[j] > m) m = row[j];

        /* Safe local copy padded to 128 lanes -- avoids reading past the
         * end of the matrix on the last row (or into the next row). Padding
         * lanes use m (diff=0, idx=255, harmless; never stored). */
        for (int j = 0; j < C; j++) xrow_buf[j] = row[j];
        for (int j = C; j < 128; j++) xrow_buf[j] = m;

        /* Step 2, vectorized over all 128 padded lanes (2 x 64-lane Vh
         * chunks): diff = sat16(x - m) (<=0 mathematically always), clamp to
         * >= -255, then +255 -> index in [0,255]. */
        HVX_Vector vm      = Q6_Vh_vsplat_R((int)m);
        HVX_Vector vNeg255 = Q6_Vh_vsplat_R(-255);
        HVX_Vector v255    = Q6_Vh_vsplat_R(255);

        HVX_Vector vX0 = *(const HVX_Vector *)(xrow_buf);
        HVX_Vector vX1 = *(const HVX_Vector *)(xrow_buf + 64);

        HVX_Vector d0 = Q6_Vh_vsub_VhVh_sat(vX0, vm);
        HVX_Vector d1 = Q6_Vh_vsub_VhVh_sat(vX1, vm);
        d0 = Q6_Vh_vmax_VhVh(d0, vNeg255);
        d1 = Q6_Vh_vmax_VhVh(d1, vNeg255);
        HVX_Vector idxH0 = Q6_Vh_vadd_VhVh(d0, v255);
        HVX_Vector idxH1 = Q6_Vh_vadd_VhVh(d1, v255);

        /* Narrow [0,255] halfwords to bytes: low-byte truncating pack is
         * exact here since the high byte of every lane is 0. */
        HVX_Vector vidx = Q6_Vb_vpacke_VhVh(idxH1, idxH0);
        *(HVX_Vector *)(idx_buf) = vidx;

        /* Steps 3-4: scalar LUT gather + sum. */
        int32_t S = 0;
        for (int j = 0; j < C; j++) {
            uint16_t e = exp_lut[idx_buf[j]];
            evals[j] = e;
            S += (int32_t)e;
        }

        /* Step 5: normalize. */
        int32_t half_S = S / 2;
        for (int j = 0; j < C; j++) {
            int64_t num = (int64_t)evals[j] * 32767 + half_S;
            orow[j] = (int16_t)(num / S);
        }
    }
}
