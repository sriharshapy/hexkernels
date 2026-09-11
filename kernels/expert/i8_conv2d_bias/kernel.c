/*
 * i8_conv2d_bias HVX v19 -- v13 + manual ky unroll for better ILP
 *
 * Unroll ky loop to expose: all 3 row loads happen before we depend on them,
 * and the vror/vrmpyacc chain can be interleaved with row loads.
 */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline int8_t requant32(int32_t acc_x, int32_t bias_co, int32_t mult, int shift, int8_t zp) {
    int32_t biased = acc_x + bias_co;
    int32_t v      = biased * mult;
    int32_t half   = (shift > 0) ? (1 << (shift - 1)) : 0;
    int32_t r;
    if (v >= 0) r = (v + half) >> shift;
    else        r = -((-v + half) >> shift);
    r += (int32_t)zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

void candidate_kernel(const int8_t *in, const int8_t *wt, const int32_t *bias,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp) {
    const int PHR = 128;
    int8_t padded[16 * 128] __attribute__((aligned(128)));
    memset(padded, 0, (H + 2) * PHR);
    for (int y = 0; y < H; y++)
        memcpy(padded + (y + 1) * PHR + C_in, in + y * W * C_in, W * C_in);

    int8_t wt_vecs[8 * 9 * 128] __attribute__((aligned(128)));
    for (int co = 0; co < C_out; co++) {
        for (int t = 0; t < 9; t++) {
            const int8_t *wtp = wt + co * 9 * C_in + t * C_in;
            int64_t *dst64 = (int64_t *)(wt_vecs + (co * 9 + t) * 128);
            int64_t wp =
                ((int64_t)(uint8_t)wtp[0]) |
                ((int64_t)(uint8_t)wtp[1] <<  8) |
                ((int64_t)(uint8_t)wtp[2] << 16) |
                ((int64_t)(uint8_t)wtp[3] << 24) |
                ((int64_t)(uint8_t)wtp[4] << 32) |
                ((int64_t)(uint8_t)wtp[5] << 40) |
                ((int64_t)(uint8_t)wtp[6] << 48) |
                ((int64_t)(uint8_t)wtp[7] << 56);
            dst64[ 0]=wp; dst64[ 1]=wp; dst64[ 2]=wp; dst64[ 3]=wp;
            dst64[ 4]=wp; dst64[ 5]=wp; dst64[ 6]=wp; dst64[ 7]=wp;
            dst64[ 8]=wp; dst64[ 9]=wp; dst64[10]=wp; dst64[11]=wp;
            dst64[12]=0;  dst64[13]=0;  dst64[14]=0;  dst64[15]=0;
        }
    }

    int32_t accbuf[32] __attribute__((aligned(128)));

    for (int co = 0; co < C_out; co++) {
        const int8_t *cw = wt_vecs + co * 9 * 128;
        int32_t bco = bias[co];
        for (int y = 0; y < H; y++) {
            /* Pre-load all 3 input rows for this y (enables pipelining with vrmpyacc) */
            const HVX_Vector *vr0p = (const HVX_Vector *)(padded + (y+0)*PHR);
            const HVX_Vector *vr1p = (const HVX_Vector *)(padded + (y+1)*PHR);
            const HVX_Vector *vr2p = (const HVX_Vector *)(padded + (y+2)*PHR);
            HVX_Vector v0 = *vr0p;
            HVX_Vector v1 = *vr1p;
            HVX_Vector v2 = *vr2p;

            /* Pre-compute rotated versions */
            HVX_Vector v0r8  = Q6_V_vror_VR(v0, 8);
            HVX_Vector v0r16 = Q6_V_vror_VR(v0, 16);
            HVX_Vector v1r8  = Q6_V_vror_VR(v1, 8);
            HVX_Vector v1r16 = Q6_V_vror_VR(v1, 16);
            HVX_Vector v2r8  = Q6_V_vror_VR(v2, 8);
            HVX_Vector v2r16 = Q6_V_vror_VR(v2, 16);

            /* Pre-load all 9 weight vectors */
            HVX_Vector vw0 = *(const HVX_Vector *)(cw + 0*128);
            HVX_Vector vw1 = *(const HVX_Vector *)(cw + 1*128);
            HVX_Vector vw2 = *(const HVX_Vector *)(cw + 2*128);
            HVX_Vector vw3 = *(const HVX_Vector *)(cw + 3*128);
            HVX_Vector vw4 = *(const HVX_Vector *)(cw + 4*128);
            HVX_Vector vw5 = *(const HVX_Vector *)(cw + 5*128);
            HVX_Vector vw6 = *(const HVX_Vector *)(cw + 6*128);
            HVX_Vector vw7 = *(const HVX_Vector *)(cw + 7*128);
            HVX_Vector vw8 = *(const HVX_Vector *)(cw + 8*128);

            /* Accumulate all 9 taps -- single dependency chain */
            HVX_Vector vacc = Q6_V_vzero();
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, v0,    vw0);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, v0r8,  vw1);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, v0r16, vw2);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, v1,    vw3);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, v1r8,  vw4);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, v1r16, vw5);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, v2,    vw6);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, v2r8,  vw7);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, v2r16, vw8);

            HVX_Vector vrot = Q6_V_vror_VR(vacc, 4);
            *(HVX_Vector *)accbuf = Q6_Vw_vadd_VwVw(vacc, vrot);

            int8_t *out_row = out + y * W * C_out;
            for (int x = 0; x < W; x++)
                out_row[x * C_out + co] = requant32(accbuf[2*x], bco, mult, shift, zp);
        }
    }
}
