/*
 * im2col HVX candidate — pinned shapes C=3,H=8,W=9,K=3,S=1,OH=6,OW=7.
 *
 * HVX strategy:
 *   1. Load vin0=in[0..127], vin1=in[128..255] (2 HVX aligned vmem loads).
 *   2. For each of 27 output rows, gather the 42 output bytes into an HVX
 *      accumulator using valign (strip fetch) + vmux (byte-window insert):
 *      - For each oh=0..5 (OH=6 strips of OW=7 bytes):
 *        a) Fetch strip: valign(vin1, vin0, src_off) or valign(vzero, vin1, src_off-128)
 *        b) Left-shift strip by P=oh*7 positions: valign(strip, vzero, 128-P)
 *        c) vmux into accumulator at [P, P+7)
 *   3. Write acc to out_buf at the row's UNALIGNED offset using HEXAGON_Vect_UN
 *      (HVX unaligned 128B store, much faster than scalar copy).
 *   4. Write out_buf to out[] in 9 HVX aligned stores (8 full + 1 masked).
 *
 * HEXAGON_Vect_UN: aligned(4) vector type that generates vmemu instructions
 * (unaligned HVX vmem load/store). This avoids the per-row scalar memcpy.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>

/* Unaligned HVX vector type: generates vmemu (128B, min 4B-aligned) */
typedef long HEXAGON_Vect_UN
    __attribute__((__vector_size__(128))) __attribute__((aligned(4)));

void candidate_kernel(const int8_t *in, int8_t *out,
                      int C, int H, int W,
                      int K, int S,
                      int OH, int OW)
{
    /* Pinned: C=3,H=8,W=9,K=3,S=1,OH=6,OW=7, row_bytes=42, total_out=1134. */
    const int row_bytes = OH * OW;     /* 42 */
    const int total_out = C*K*K*OH*OW; /* 1134 = 8*128 + 110 */

    /* HVX aligned loads: 2 vmem instructions cover the 216-byte input.
     * in is HVX_ALIGN (128B-aligned) per harness.  vp[1]=in+128 may touch
     * in[216..255] (beyond array), but those bytes are never used (max src_off+6=215). */
    HVX_Vector vin0 = *(const HVX_Vector *)in;
    HVX_Vector vin1 = *(const HVX_Vector *)(in + 128);
    HVX_Vector vzero = Q6_V_vzero();

    /* Aligned output staging buffer (1152B = 9*128B ≥ 1134B). */
    int8_t out_buf[1152] __attribute__((aligned(128)));

    /* Predicate for first strip (P=0): covers bytes 0..OW-1 */
    HVX_VectorPred q_ow = Q6_Q_vsetq2_R(OW);

    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                int row = c*K*K + kh*K + kw;
                /* out_buf + row*42 is the (possibly unaligned) output row start */

                HVX_Vector acc = vzero;

                for (int oh = 0; oh < OH; oh++) {
                    /* Source offset in the 216-byte input */
                    int src_off = c*(H*W) + (oh*S + kh)*W + kw;
                    int P = oh * OW;   /* output byte position of this strip: 0,7,14,21,28,35 */

                    /* Fetch 7-byte strip aligned to byte 0 of strip_v.
                     * valign(hi, lo, R): result[0] = lo[R] (R taken mod 128 by HW).
                     * For src_off<128: lo=vin0, hi=vin1, R=src_off → result[0]=in[src_off].
                     * For src_off>=128: lo=vin1, hi=vzero, R=src_off-128 → result[0]=in[src_off]. */
                    HVX_Vector strip_v;
                    if (src_off < 128) {
                        strip_v = Q6_V_valign_VVR(vin1, vin0, src_off);
                    } else {
                        strip_v = Q6_V_valign_VVR(vzero, vin1, src_off - 128);
                    }

                    /* Insert strip_v[0..OW-1] into accumulator at bytes [P, P+OW).
                     *
                     * Left-shift strip_v by P: valign(strip_v, vzero, 128-P)
                     *   → result[P+j] = strip_v[j], result[0..P-1] = 0
                     *
                     * Window-insert:
                     *   saved = acc
                     *   acc = vmux(vsetq2(P+OW), shifted, acc)  [bytes 0..P+OW-1 from shifted]
                     *   acc = vmux(vsetq2(P),    saved,   acc)   [bytes 0..P-1 restored from saved]
                     */
                    if (P == 0) {
                        /* Strip goes at byte 0: no shift, one vmux */
                        acc = Q6_V_vmux_QVV(q_ow, strip_v, acc);
                    } else {
                        HVX_Vector shifted = Q6_V_valign_VVR(strip_v, vzero, 128 - P);
                        HVX_Vector saved   = acc;
                        acc = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(P + OW), shifted, acc);
                        acc = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(P),       saved,   acc);
                    }
                }

                /* Write acc to out_buf using unaligned HVX store (vmemu).
                 * The destination is row*42 bytes into out_buf; alignment varies. */
                *((HEXAGON_Vect_UN *)(out_buf + row * row_bytes)) = (HEXAGON_Vect_UN)acc;
            }
        }
    }

    /* Copy assembled out_buf → out[] using 8 aligned HVX stores + 1 masked tail.
     * Both buffers are 128B-aligned. total_out = 1134 = 8*128 + 110. */
    HVX_Vector *sv = (HVX_Vector *)out_buf;
    HVX_Vector *dv = (HVX_Vector *)out;
    dv[0] = sv[0]; dv[1] = sv[1]; dv[2] = sv[2]; dv[3] = sv[3];
    dv[4] = sv[4]; dv[5] = sv[5]; dv[6] = sv[6]; dv[7] = sv[7];
    Q6_vmem_QRIV(Q6_Q_vsetq2_R(total_out - 8*128), dv + 8, sv[8]);
}
