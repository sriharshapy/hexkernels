/*
 * i8_depth_to_space — HVX kernel
 * Pinned: C_in=12, H_in=4, W_in=5, b=2
 * C_out=3, H_out=8, W_out=10
 *
 * HVX strategy (adapted from i8_im2col pattern):
 *   1. Two aligned HVX vmem loads: V_in0=in[0..127], V_in1=in[128..255].
 *   2. For each of 24 output rows (3 channels × 8 rows):
 *      a. valign to extract 5 bytes from even-channel (bw=0).
 *      b. valign to extract 5 bytes from odd-channel (bw=1).
 *      c. Q6_W_vshuff_VVR(vodd, veven, -1): byte-level interleave → 10B row.
 *      d. HEXAGON_Vect_UN store to out_buf at row offset (vmemu unaligned store).
 *         128 bytes are written but only first 10 matter; garbage is overwritten
 *         by subsequent rows' stores (rows in order 0,1,...,23).
 *   3. 2 aligned HVX vector stores: out_buf[0..127] → out[0..127],
 *                                     out_buf[128..255] → out[128..255].
 *
 * HVX idiom: valign for input gather, Q6_W_vshuff_VVR(-1) for byte interleave,
 *            HEXAGON_Vect_UN for non-128-aligned row stores.
 */

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>

/* Unaligned HVX vector type: generates vmemu instructions (handles non-128-aligned) */
typedef long HEXAGON_Vect_UN
    __attribute__((__vector_size__(128))) __attribute__((aligned(4)));

void candidate_kernel(const int8_t *in, int8_t *out,
                      int C_in, int H_in, int W_in, int b) {
    const int C_out    = C_in / (b * b);   /* 3 */
    const int H_out    = H_in * b;          /* 8 */
    const int W_out    = W_in * b;          /* 10 */
    const int plane_in = H_in * W_in;       /* 20 */
    const int plane_out= H_out * W_out;     /* 80 */
    (void)H_out;

    /*
     * Staging buffer: needs to hold up to out_buf[230+127]=out_buf[357], so 384 bytes.
     * 384 = 3 * 128.
     */
    int8_t out_buf[384] __attribute__((aligned(128)));

    /* 2 aligned HVX loads covering all 240 input bytes */
    HVX_Vector vin0 = *(const HVX_Vector *)in;          /* in[0..127] */
    HVX_Vector vin1 = *(const HVX_Vector *)(in + 128);  /* in[128..255], 112 valid */
    HVX_Vector vzero = Q6_V_vzero();

    /*
     * For each output row (row index = c*8 + oh*2 + bh):
     *   source even-channel offset: (c*4 + bh*2 + 0) * 20 + oh*5
     *   source odd-channel offset:  (c*4 + bh*2 + 1) * 20 + oh*5
     */
    for (int c = 0; c < C_out; c++) {
        int base_cin = c * b * b;           /* 0, 4, 8 */

        /* Process rows in ASCENDING dst_off order: c→oh→bh
         * Row at dst_off=X writes 128 bytes starting at X.
         * By ascending order, garbage bytes [10..127] from row X
         * are overwritten by correct data from rows X+10, X+20, ... */
        for (int oh = 0; oh < H_in; oh++) {
            for (int bh = 0; bh < b; bh++) {
            int c_in_e = base_cin + bh * b + 0;  /* bw=0 */
            int c_in_o = base_cin + bh * b + 1;  /* bw=1 */

                int off_e = c_in_e * plane_in + oh * W_in;
                int off_o = c_in_o * plane_in + oh * W_in;

                /* Extract even-channel bytes via valign.
                 * valign(Vu, Vv, R): result[k] = Vv[k+R] if k+R<128 else Vu[k+R-128].
                 * For off < 128: Vv=vin0, Vu=vin1, R=off → result[k]=in[off+k]
                 * For off >= 128: Vv=vin1, Vu=vzero, R=off-128 → result[k]=in[off+k] */
                HVX_Vector veven = (off_e < 128)
                    ? Q6_V_valign_VVR(vin1, vin0, off_e)
                    : Q6_V_valign_VVR(vzero, vin1, off_e - 128);

                HVX_Vector vodd = (off_o < 128)
                    ? Q6_V_valign_VVR(vin1, vin0, off_o)
                    : Q6_V_valign_VVR(vzero, vin1, off_o - 128);

                /* Byte-interleave even and odd:
                 * Q6_W_vshuff_VVR(Vu=vodd, Vv=veven, Rt=-1):
                 *   lo[2i]   = veven[i]  (bw=0: even output columns)
                 *   lo[2i+1] = vodd[i]   (bw=1: odd output columns)
                 * Result: lo[0..9] = interleaved output row */
                HVX_VectorPair vp  = Q6_W_vshuff_VVR(vodd, veven, -1);
                HVX_Vector     vrow = Q6_V_lo_W(vp);

                /* Write 128-byte vector to out_buf at row's byte offset (unaligned store).
                 * Rows processed in order 0,1,...,23; garbage bytes from each store
                 * are overwritten by subsequent rows' stores.
                 * Row index in output: c*8 + oh*b + bh (but we write to c*plane_out + (oh*b+bh)*W_out). */
                int dst_off = c * plane_out + (oh * b + bh) * W_out;
                *((HEXAGON_Vect_UN *)(out_buf + dst_off)) = (HEXAGON_Vect_UN)vrow;
            } /* bh */
        } /* oh */
    } /* c */

    /* Flush out_buf[0..239] → out using 2 aligned HVX vector stores.
     * out is HVX_ALIGN; out[128..255] store covers out[128..239] valid + [240..255] padding. */
    *(HVX_Vector *)out        = *(HVX_Vector *)out_buf;
    *(HVX_Vector *)(out + 128) = *(HVX_Vector *)(out_buf + 128);
}
