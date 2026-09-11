/*
 * i8_conv1d_dilated_requant HVX kernel -- expert
 *
 * out[i] = requant( sum_{j=0..ntaps-1} x[i + j*dilation] * taps[j] ), i in [0,n).
 * VALID (no padding). n=512, ntaps=7, dilation runtime-swept {1,2,3}. Requant:
 * round-half-away-from-zero then +zp then saturate to int8.
 *
 * Key insight: dilation only changes the BYTE OFFSET between taps, not
 * between output positions -- for a fixed tap j, the 128 (or fewer)
 * consecutive output positions i0..i0+127 read x[i+j*dilation], which is
 * still a CONTIGUOUS 128-byte vector load (just starting j*dilation bytes
 * further into x). So the depthwise/SAME per-tap widen-accumulate idiom
 * applies unchanged: for each tap j, load 128B at x+i0+j*dilation
 * (unaligned), sign-widen to int16, multiply by broadcast scalar tap,
 * widen to int32, accumulate. Then vectorized requantize (scalar mult/shift
 * broadcast across the 128-lane accumulator, done in the int32 domain).
 *
 * Defining MAC: Q6_Vh_vmpyi_VhVh (real HVX vector multiply).
 */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int n, int ntaps, int dilation,
                      int32_t mult, int shift, int8_t zp) {
    const int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    const int32_t izp = (int32_t)zp;

    int32_t accbuf[128] __attribute__((aligned(128)));

    for (int i0 = 0; i0 < n; i0 += 128) {
        int chunk = (n - i0 < 128) ? (n - i0) : 128;

        HVX_Vector acc_w0 = Q6_V_vzero();
        HVX_Vector acc_w1 = Q6_V_vzero();
        HVX_Vector acc_w2 = Q6_V_vzero();
        HVX_Vector acc_w3 = Q6_V_vzero();

        for (int j = 0; j < ntaps; j++) {
            HVX_Vector vin = *(const HVX_UVector *)(x + i0 + j * dilation);
            HVX_VectorPair iw = Q6_Wh_vunpack_Vb(vin); /* sequential: lo=0..63,hi=64..127 */
            HVX_Vector ilo = Q6_V_lo_W(iw);
            HVX_Vector ihi = Q6_V_hi_W(iw);

            int16_t tv = (int16_t)taps[j];
            HVX_Vector vtap = Q6_Vh_vsplat_R((int32_t)tv);

            HVX_Vector plo = Q6_Vh_vmpyi_VhVh(ilo, vtap);
            HVX_Vector phi = Q6_Vh_vmpyi_VhVh(ihi, vtap);

            HVX_VectorPair wplo = Q6_Ww_vunpack_Vh(plo);
            HVX_VectorPair wphi = Q6_Ww_vunpack_Vh(phi);

            acc_w0 = Q6_Vw_vadd_VwVw(acc_w0, Q6_V_lo_W(wplo));
            acc_w1 = Q6_Vw_vadd_VwVw(acc_w1, Q6_V_hi_W(wplo));
            acc_w2 = Q6_Vw_vadd_VwVw(acc_w2, Q6_V_lo_W(wphi));
            acc_w3 = Q6_Vw_vadd_VwVw(acc_w3, Q6_V_hi_W(wphi));
        }

        *(HVX_Vector *)(accbuf +  0) = acc_w0;
        *(HVX_Vector *)(accbuf + 32) = acc_w1;
        *(HVX_Vector *)(accbuf + 64) = acc_w2;
        *(HVX_Vector *)(accbuf + 96) = acc_w3;

        for (int i = 0; i < chunk; i++) {
            int64_t v = (int64_t)accbuf[i] * (int64_t)mult;
            int64_t r;
            if (v >= 0) r = (v + half) >> shift;
            else        r = -((-v + half) >> shift);
            r += izp;
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            out[i0 + i] = (int8_t)r;
        }
    }
}
