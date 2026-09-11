/* NCHW->NHWC  C=4, H=10, W=13  N_ELEM=520
 * Pinned: HW=130 spatial positions, 4 channels.
 * Output: 4 full 128-byte vectors (output positions 0..127, 128..255, 256..383, 384..511)
 *         + 8-byte tail (output positions 512..519, HW positions 128-129).
 *
 * Strategy: load each channel's 130 bytes from vmem (128-aligned reads + valign),
 * then do 4-way byte interleave with Q6_W_vshuff_VVR to produce 128-byte output chunks.
 *
 * For each output group of 32 HW positions (128 bytes out):
 *   gather 32 bytes each from ch0,ch1,ch2,ch3 starting at hw_base
 *   interleave: vshuff(cb,ca,-1) -> pairs, vshuff(pairs_CD,pairs_AB,-2) -> quads
 */

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>

/* Unaligned 128-byte HVX load from arbitrary byte pointer */
static inline HVX_Vector hvx_load_unaligned(const int8_t *ptr) {
    const HVX_Vector *p0 = (const HVX_Vector *)(((uintptr_t)ptr) & ~127UL);
    HVX_Vector lo = *p0;
    HVX_Vector hi = *(p0 + 1);
    return Q6_V_valign_VVR(hi, lo, (int)((uintptr_t)ptr & 127));
}

void candidate_kernel(const int8_t *in, int8_t *out, int C, int H, int W) {
    (void)C; (void)H; (void)W;
    /* Pinned: C=4, H=10, W=13, HW=130 */
    const int HW = 130;
    const int8_t *ch0 = in + 0 * HW;  /* bytes 0..129   */
    const int8_t *ch1 = in + 1 * HW;  /* bytes 130..259 */
    const int8_t *ch2 = in + 2 * HW;  /* bytes 260..389 */
    const int8_t *ch3 = in + 3 * HW;  /* bytes 390..519 */
    HVX_Vector *dst = (HVX_Vector *)out; /* out is HVX_ALIGN */

    /* Load full 128-byte vectors from each channel (channels are 130 bytes, slightly > 1 vec).
     * We'll load 2 vecs per channel to cover all 130 bytes, then extract 32-byte slices. */

    /* ch0: starts at in+0 (aligned), ch0[0..127] in vch0_0, ch0[128..129] in vch0_1 */
    const HVX_Vector *p0 = (const HVX_Vector *)ch0;  /* aligned */
    HVX_Vector vch0_0 = p0[0];   /* ch0[0..127]  */
    HVX_Vector vch0_1 = p0[1];   /* ch0[128..255] but only [128..129] valid */

    /* ch1: starts at in+130, alignment offset = 130 & 127 = 2 */
    HVX_Vector vch1_0 = hvx_load_unaligned(ch1);       /* ch1[0..127]  */
    HVX_Vector vch1_1 = hvx_load_unaligned(ch1 + 128); /* ch1[128..129] in [0..1] */

    /* ch2: starts at in+260, alignment offset = 260 & 127 = 4 */
    HVX_Vector vch2_0 = hvx_load_unaligned(ch2);
    HVX_Vector vch2_1 = hvx_load_unaligned(ch2 + 128);

    /* ch3: starts at in+390, alignment offset = 390 & 127 = 6 */
    HVX_Vector vch3_0 = hvx_load_unaligned(ch3);
    HVX_Vector vch3_1 = hvx_load_unaligned(ch3 + 128);

    /* Output group 0: HW positions 0..31 → output bytes 0..127
     * Use low 32 bytes of each channel vector (positions 0..31) */
    {
        /* vshuff(Vu, Vv, -1): lo half = interleaved byte pairs [Vv[0],Vu[0],Vv[1],Vu[1],...]
         * We need [c0[hw],c1[hw],c2[hw],c3[hw]] for hw=0..31
         * The 32 bytes of interest are in the LOW 32 bytes of each vchN_0
         * After vshuff(...,-1), the 32 pairs occupy bytes 0..63 of the lo vector */
        HVX_VectorPair W_AB = Q6_W_vshuff_VVR(vch1_0, vch0_0, -1);
        HVX_VectorPair W_CD = Q6_W_vshuff_VVR(vch3_0, vch2_0, -1);
        /* lo of W_AB: [a0,b0,a1,b1,...,a63,b63] — pairs for hw 0..63
         * lo of W_CD: [c0,d0,c1,d1,...,c63,d63] — pairs for hw 0..63
         * vshuff with -2 (halfword): interleaves 2-byte elements
         * lo of result: [a0,b0,c0,d0, a1,b1,c1,d1, ..., a31,b31,c31,d31] */
        HVX_VectorPair W_ABCD = Q6_W_vshuff_VVR(Q6_V_lo_W(W_CD), Q6_V_lo_W(W_AB), -2);
        dst[0] = Q6_V_lo_W(W_ABCD);
    }

    /* Output group 1: HW positions 32..63 → output bytes 128..255
     * Use bytes 32..63 of each vchN_0, which are in bits 256..511 of the vector.
     * Extract by rotating/aligning: valign(vch0_0, vch0_0, 32) rotates left 32 bytes */
    {
        HVX_Vector a32 = Q6_V_valign_VVR(vch0_0, vch0_0, 32);
        HVX_Vector b32 = Q6_V_valign_VVR(vch1_0, vch1_0, 32);
        HVX_Vector c32 = Q6_V_valign_VVR(vch2_0, vch2_0, 32);
        HVX_Vector d32 = Q6_V_valign_VVR(vch3_0, vch3_0, 32);
        HVX_VectorPair W_AB = Q6_W_vshuff_VVR(b32, a32, -1);
        HVX_VectorPair W_CD = Q6_W_vshuff_VVR(d32, c32, -1);
        HVX_VectorPair W_ABCD = Q6_W_vshuff_VVR(Q6_V_lo_W(W_CD), Q6_V_lo_W(W_AB), -2);
        dst[1] = Q6_V_lo_W(W_ABCD);
    }

    /* Output group 2: HW positions 64..95 → output bytes 256..383 */
    {
        HVX_Vector a64 = Q6_V_valign_VVR(vch0_0, vch0_0, 64);
        HVX_Vector b64 = Q6_V_valign_VVR(vch1_0, vch1_0, 64);
        HVX_Vector c64 = Q6_V_valign_VVR(vch2_0, vch2_0, 64);
        HVX_Vector d64 = Q6_V_valign_VVR(vch3_0, vch3_0, 64);
        HVX_VectorPair W_AB = Q6_W_vshuff_VVR(b64, a64, -1);
        HVX_VectorPair W_CD = Q6_W_vshuff_VVR(d64, c64, -1);
        HVX_VectorPair W_ABCD = Q6_W_vshuff_VVR(Q6_V_lo_W(W_CD), Q6_V_lo_W(W_AB), -2);
        dst[2] = Q6_V_lo_W(W_ABCD);
    }

    /* Output group 3: HW positions 96..127 → output bytes 384..511 */
    {
        HVX_Vector a96 = Q6_V_valign_VVR(vch0_0, vch0_0, 96);
        HVX_Vector b96 = Q6_V_valign_VVR(vch1_0, vch1_0, 96);
        HVX_Vector c96 = Q6_V_valign_VVR(vch2_0, vch2_0, 96);
        HVX_Vector d96 = Q6_V_valign_VVR(vch3_0, vch3_0, 96);
        HVX_VectorPair W_AB = Q6_W_vshuff_VVR(b96, a96, -1);
        HVX_VectorPair W_CD = Q6_W_vshuff_VVR(d96, c96, -1);
        HVX_VectorPair W_ABCD = Q6_W_vshuff_VVR(Q6_V_lo_W(W_CD), Q6_V_lo_W(W_AB), -2);
        dst[3] = Q6_V_lo_W(W_ABCD);
    }

    /* Tail: HW positions 128..129 → output bytes 512..519 (8 bytes) */
    {
        /* vch0_1[0]=ch0[128], vch0_1[1]=ch0[129], similarly for others */
        /* Use scalar for 2-element tail */
        int8_t *t = (int8_t *)(dst + 4);
        /* Extract bytes 0 and 1 from tail vectors */
        int8_t tail_buf[128] __attribute__((aligned(128)));
        *(HVX_Vector *)tail_buf = vch0_1;
        t[0] = tail_buf[0]; t[4] = tail_buf[1];
        *(HVX_Vector *)tail_buf = vch1_1;
        t[1] = tail_buf[0]; t[5] = tail_buf[1];
        *(HVX_Vector *)tail_buf = vch2_1;
        t[2] = tail_buf[0]; t[6] = tail_buf[1];
        *(HVX_Vector *)tail_buf = vch3_1;
        t[3] = tail_buf[0]; t[7] = tail_buf[1];
    }
}
