/*
 * HVX matmul+softmax tile: SEQ_Q=16, SEQ_K=16, HEAD_DIM=32.
 *
 * Step A: raw[i,j] = sum_d Q[i,d]*K[j,d]  (int32) -> fixed-point scale by
 *         inv_sqrt_d>>shift (round-half-up) -> clamp to int8 scores.
 * Step B: rowwise softmax over int8 scores via runtime exp_lut -> uint8 out.
 *
 * HVX strategy for Step A (the defining matmul compute): same vrmpy-tile idiom
 * as i8_gemm_tile / attention_qkt_tile, but with a 2-QUERY-ROWS-PER-VRMPY packing
 * so all 32 HVX accumulator lanes do useful work (SEQ_K=16 alone only fills half
 * the vector):
 *   - Pre-pack K into Kt[kg*128 + j*4 + r] = K[j*HEAD_DIM + kg*4 + r] for j in
 *     [0,16), duplicated into bytes [64:128) as well (kg*128+64+j*4+r), so a
 *     32-lane accumulator vector can host two independent query rows.
 *   - Per row-pair (i0,i1): build va lane j<16 = Q[i0][k0..k0+3] broadcast,
 *     lane j>=16 = Q[i1][k0..k0+3] broadcast, via two Q6_V_vsplat_R + one
 *     Q6_V_valign_VVR half-swap (in-register, no memory round trip).
 *   - vrmpyacc(acc, va, vb): acc[j<16] += Q[i0].K[j], acc[j>=16] += Q[i1].K[j-16].
 *
 * Step B (softmax) is scalar -- LUT lookup has no HVX gather in the bare sim
 * (vgather faults, no VTCM mapping) and the defining compute here is the matmul.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

#define SEQ_Q    16
#define SEQ_K    16
#define HEAD_DIM 32

void candidate_kernel(const int8_t *Q, const int8_t *K,
                      uint8_t *out,
                      const uint8_t *exp_lut,
                      int32_t inv_sqrt_d, int shift)
{
    enum { NKG = HEAD_DIM / 4 }; /* 32/4 = 8, exact, no tail */

    /* Kt: NKG x 128 bytes. Lanes 0..15 and 16..31 both hold the same
     * K[j][k0..k0+3] (j=0..15) so a 2-row-packed Q vector produces useful
     * dot products in both halves. */
    static int8_t __attribute__((aligned(128))) Kt[NKG * 128];

    for (int kg = 0; kg < NKG; kg++) {
        int8_t *blk = Kt + kg * 128;
        int k0 = kg * 4;
        for (int j = 0; j < SEQ_K; j++) {
            int8_t b0 = K[j*HEAD_DIM + k0 + 0];
            int8_t b1 = K[j*HEAD_DIM + k0 + 1];
            int8_t b2 = K[j*HEAD_DIM + k0 + 2];
            int8_t b3 = K[j*HEAD_DIM + k0 + 3];
            blk[j*4+0] = b0; blk[j*4+1] = b1; blk[j*4+2] = b2; blk[j*4+3] = b3;
            blk[64 + j*4+0] = b0; blk[64 + j*4+1] = b1; blk[64 + j*4+2] = b2; blk[64 + j*4+3] = b3;
        }
        /* SEQ_K == 16 exactly fills the 16 word-lanes per half; no padding needed. */
    }

    /* Precompute per-row 4-byte-packed Q words for all k-groups. */
    uint32_t qwords[SEQ_Q][NKG];
    for (int i = 0; i < SEQ_Q; i++) {
        const int8_t *Qrow = Q + i * HEAD_DIM;
        for (int kg = 0; kg < NKG; kg++) {
            int k0 = kg * 4;
            qwords[i][kg] = (uint32_t)((uint8_t)Qrow[k0+0])
                          | ((uint32_t)((uint8_t)Qrow[k0+1]) << 8)
                          | ((uint32_t)((uint8_t)Qrow[k0+2]) << 16)
                          | ((uint32_t)((uint8_t)Qrow[k0+3]) << 24);
        }
    }

    int64_t half_val = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0LL;

    int8_t scores[SEQ_Q * SEQ_K] __attribute__((aligned(128)));
    int32_t accbuf[32] __attribute__((aligned(128)));

    for (int ip = 0; ip < SEQ_Q; ip += 2) {
        int i0 = ip, i1 = ip + 1;
        HVX_Vector acc = Q6_V_vzero();

        for (int kg = 0; kg < NKG; kg++) {
            uint32_t w0 = qwords[i0][kg];
            uint32_t w1 = qwords[i1][kg];
            HVX_Vector vA = Q6_V_vsplat_R(w0);
            HVX_Vector vB = Q6_V_vsplat_R(w1);
            /* Q6_V_valign_VVR(Vu,Vv,Rt): concat={Vu hi128B :: Vv lo128B}, window
             * @byte-offset Rt. Q6_V_valign_VVR(vB,vA,64): concat={vB,vA}; window@64
             * = vA[64:128) :: vB[0:64). vA/vB are splats (uniform 4B pattern), so
             * any 64B half reproduces the pattern -> result[0:64)=vA pattern (i0),
             * result[64:128)=vB pattern (i1). */
            HVX_Vector va = Q6_V_valign_VVR(vB, vA, 64);
            HVX_Vector vb = *(const HVX_Vector *)(Kt + kg * 128);
            acc = Q6_Vw_vrmpyacc_VwVbVb(acc, va, vb);
        }

        *(HVX_Vector *)accbuf = acc;

        int8_t *srow0 = scores + i0 * SEQ_K;
        int8_t *srow1 = scores + i1 * SEQ_K;
        for (int j = 0; j < SEQ_K; j++) {
            int64_t v = (int64_t)accbuf[j] * (int64_t)inv_sqrt_d;
            int64_t sc = (v >= 0) ? ((v + half_val) >> shift)
                                   : -((-v + half_val) >> shift);
            if (sc >  127) sc =  127;
            if (sc < -128) sc = -128;
            srow0[j] = (int8_t)sc;
        }
        for (int j = 0; j < SEQ_K; j++) {
            int64_t v = (int64_t)accbuf[16 + j] * (int64_t)inv_sqrt_d;
            int64_t sc = (v >= 0) ? ((v + half_val) >> shift)
                                   : -((-v + half_val) >> shift);
            if (sc >  127) sc =  127;
            if (sc < -128) sc = -128;
            srow1[j] = (int8_t)sc;
        }
    }

    /* Step B: rowwise softmax (scalar; LUT indexing has no HVX vgather path
     * available in the bare sim). */
    for (int i = 0; i < SEQ_Q; i++) {
        const int8_t *row = scores + i * SEQ_K;
        uint8_t      *orow = out   + i * SEQ_K;

        int8_t m = row[0];
        for (int j = 1; j < SEQ_K; j++) if (row[j] > m) m = row[j];

        int32_t S = 0;
        for (int j = 0; j < SEQ_K; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            orow[j] = e;
            S += (int32_t)e;
        }

        int32_t half_S = S / 2;
        for (int j = 0; j < SEQ_K; j++) {
            int32_t ej = (int32_t)orow[j];
            orow[j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}
