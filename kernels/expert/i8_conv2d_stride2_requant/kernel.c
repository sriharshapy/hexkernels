/*
 * i8_conv2d_stride2_requant HVX kernel — expert
 *
 * Shapes (from harness): H=W=16, C_in=C_out=8, filter=3x3, stride=2, SAME pad.
 * Output: out_H=out_W=8.
 *
 * Defining MAC: Q6_Vw_vrmpyacc_VwVbVb (signed int8x int8 4-wide dot, 32 lanes/vector).
 *
 * Idiom (same as i8_conv2d_bias, adapted for stride=2):
 *   Each padded input row is exactly 128B (16 pixels * C_in=8) -- fills ONE HVX
 *   vector with NO room for left/right zero columns inside the row itself.
 *   Left/right SAME-pad (pixel -1, pixel 16) is realized by zeroing 8 bytes at
 *   each end of a 144B row buffer and loading a vector at an 8-byte offset that
 *   straddles those pad bytes (unaligned vmem load), exactly like conv2d_bias's
 *   vror trick but via offset load instead of rotate (row is full-width so vror
 *   would wrap around instead of shifting in zeros).
 *
 *   For each of the 3 kx taps we load the row at byte-offset (kx)*C_in relative
 *   to the padded row start (kx=0 -> offset 0 covers pixels [-1..14],
 *   kx=1 -> offset 8 covers pixels [0..15], kx=2 -> offset 16 covers [1..16]).
 *   vrmpyacc against the correspondingly-shifted weight-broadcast vector gives,
 *   in pair-summed lane x (x=0..15), the partial dot for input column x.
 *   We only need EVERY OTHER x (x = 2*ox) for stride-2 output, so we read
 *   accbuf[2*(2*ox)] = accbuf[4*ox] after pair-summing.
 *
 * 3-way independent vrmpyacc chains (one per kx) hide vrmpyacc's 4-cycle
 * latency, mirroring i8_conv2d_bias.
 */

#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, const int8_t *wt,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp) {
    const int PW = W * C_in;      /* pixels-bytes per row = 128 for W=16,C_in=8 */
    const int ROW = PW + 16;      /* 8B left pad + PW + 8B right pad */

    /*
     * Padded buffer: (H+2) rows x ROW bytes.
     * Row (y+1) holds input row y, at byte offset 8 (after left pad).
     * Row 0 and row H+1 are all-zero (top/bottom SAME pad).
     * Left 8 bytes and right 8 bytes of every row are zero (left/right pad).
     */
    int8_t padded[18 * 144] __attribute__((aligned(128)));
    memset(padded, 0, (size_t)(H + 2) * ROW);
    for (int y = 0; y < H; y++)
        memcpy(padded + (y + 1) * ROW + C_in, in + y * PW, PW);

    /*
     * Weight vectors: for each (co, tap t=0..8), a 128-byte vector.
     * Bytes 0..127 = 8-byte weight pattern [w0..w7] replicated 16 times
     * (W=16 pixel positions all get the same per-channel weight taps).
     */
    int8_t wt_vecs[8 * 9 * 128] __attribute__((aligned(128)));
    for (int co = 0; co < C_out; co++) {
        for (int t = 0; t < 9; t++) {
            const int8_t *wtp = wt + co * 9 * C_in + t * C_in;
            int64_t *dst64 = (int64_t *)(wt_vecs + (co * 9 + t) * 128);
            int64_t wp =
                ((int64_t)(uint8_t)wtp[0])        |
                ((int64_t)(uint8_t)wtp[1] <<  8)  |
                ((int64_t)(uint8_t)wtp[2] << 16)  |
                ((int64_t)(uint8_t)wtp[3] << 24)  |
                ((int64_t)(uint8_t)wtp[4] << 32)  |
                ((int64_t)(uint8_t)wtp[5] << 40)  |
                ((int64_t)(uint8_t)wtp[6] << 48)  |
                ((int64_t)(uint8_t)wtp[7] << 56);
            for (int i = 0; i < 16; i++) dst64[i] = wp;
        }
    }

    const int32_t half = (shift > 0) ? (1 << (shift - 1)) : 0;
    const int32_t izp  = (int32_t)zp;

    const int out_H = H / 2;
    const int out_W = W / 2;

    int32_t accbuf[32] __attribute__((aligned(128)));

    for (int co = 0; co < C_out; co++) {
        const int8_t *cw = wt_vecs + co * 9 * 128;

        for (int oy = 0; oy < out_H; oy++) {
            int iy_base = oy * 2;           /* stride=2 */
            /* padded row index for input row (iy_base+ky-1) is (iy_base+ky-1)+1
               = iy_base+ky, for ky=0,1,2 -> rows iy_base, iy_base+1, iy_base+2 */
            const int8_t *r0 = padded + (iy_base + 0) * ROW;
            const int8_t *r1 = padded + (iy_base + 1) * ROW;
            const int8_t *r2 = padded + (iy_base + 2) * ROW;

            /* kx=0,1,2 -> byte offsets 0,8,16 within the row (unaligned loads
               since ROW is not a multiple of 128 and offsets are 8B-granular). */
            HVX_Vector v0_k0 = *(const HVX_UVector *)(r0 + 0);
            HVX_Vector v0_k1 = *(const HVX_UVector *)(r0 + 8);
            HVX_Vector v0_k2 = *(const HVX_UVector *)(r0 + 16);
            HVX_Vector v1_k0 = *(const HVX_UVector *)(r1 + 0);
            HVX_Vector v1_k1 = *(const HVX_UVector *)(r1 + 8);
            HVX_Vector v1_k2 = *(const HVX_UVector *)(r1 + 16);
            HVX_Vector v2_k0 = *(const HVX_UVector *)(r2 + 0);
            HVX_Vector v2_k1 = *(const HVX_UVector *)(r2 + 8);
            HVX_Vector v2_k2 = *(const HVX_UVector *)(r2 + 16);

            HVX_Vector vw0 = *(const HVX_Vector *)(cw + 0*128);
            HVX_Vector vw1 = *(const HVX_Vector *)(cw + 1*128);
            HVX_Vector vw2 = *(const HVX_Vector *)(cw + 2*128);
            HVX_Vector vw3 = *(const HVX_Vector *)(cw + 3*128);
            HVX_Vector vw4 = *(const HVX_Vector *)(cw + 4*128);
            HVX_Vector vw5 = *(const HVX_Vector *)(cw + 5*128);
            HVX_Vector vw6 = *(const HVX_Vector *)(cw + 6*128);
            HVX_Vector vw7 = *(const HVX_Vector *)(cw + 7*128);
            HVX_Vector vw8 = *(const HVX_Vector *)(cw + 8*128);

            /* 3-way independent chains: va=kx0 taps(0,3,6), vb=kx1 taps(1,4,7),
               vc=kx2 taps(2,5,8). */
            HVX_Vector va = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), v0_k0, vw0);
            HVX_Vector vb = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), v0_k1, vw1);
            HVX_Vector vc = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), v0_k2, vw2);

            va = Q6_Vw_vrmpyacc_VwVbVb(va, v1_k0, vw3);
            vb = Q6_Vw_vrmpyacc_VwVbVb(vb, v1_k1, vw4);
            vc = Q6_Vw_vrmpyacc_VwVbVb(vc, v1_k2, vw5);

            va = Q6_Vw_vrmpyacc_VwVbVb(va, v2_k0, vw6);
            vb = Q6_Vw_vrmpyacc_VwVbVb(vb, v2_k1, vw7);
            vc = Q6_Vw_vrmpyacc_VwVbVb(vc, v2_k2, vw8);

            HVX_Vector vacc = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(va, vb), vc);

            /* Pair-sum: lane 2x + lane 2x+1 = acc for pixel x (x=0..15). */
            HVX_Vector vrot = Q6_V_vror_VR(vacc, 4);
            *(HVX_Vector *)accbuf = Q6_Vw_vadd_VwVw(vacc, vrot);

            /* Stride-2: only pixels x = 2*ox (ox=0..out_W-1) are wanted. */
            int8_t *out_row = out + oy * out_W * C_out;
            for (int ox = 0; ox < out_W; ox++) {
                int x = 2 * ox;
                int32_t v = accbuf[2*x] * mult;
                int32_t r;
                if (v >= 0) r = (v + half) >> shift;
                else        r = -((-v + half) >> shift);
                r += izp;
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                out_row[ox * C_out + co] = (int8_t)r;
            }
        }
    }
}
