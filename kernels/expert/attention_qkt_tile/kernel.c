/*
 * HVX QK^T tile v2: M=16, N=16, D=130.
 *
 * raw[i,j] = sum_d Q[i,d]*K[j,d]; then int64 round-half-away-from-zero requant to int8.
 *
 * Key improvement over the "one row per vrmpy, 16/32 lanes used" baseline (3.7x):
 * pack TWO query rows per vrmpy call so all 32 accumulator lanes are useful work
 * (lanes 0..15 = row i0 . K[0..15], lanes 16..31 = row i1 . K[0..15]). This halves
 * the number of vrmpy-bearing outer iterations (8 row-pairs instead of 16 rows).
 *
 * Packing:
 *   Kt[kg*128 + j*4 + r]        = K[j*D + kg*4 + r]      for j in [0,16)   (lanes 0..15)
 *   Kt[kg*128 + 64 + j*4 + r]   = K[j*D + kg*4 + r]      for j in [0,16)   (lanes 16..31, same K, repeated)
 * (One 128B block per k-group covers both row-halves identically -- reused across all M/2 row-pairs.)
 *
 * Per row-pair (i0,i1), per k-group kg:
 *   va lane j<16  = Q[i0][k0..k0+3]  (same 4 bytes broadcast to lanes 0..15)
 *   va lane j>=16 = Q[i1][k0..k0+3]  (same 4 bytes broadcast to lanes 16..31)
 *   vrmpyacc(acc, va, vb) -> acc[j<16] += Q[i0].K[j], acc[j>=16] += Q[i1].K[j-16]
 *
 * va is built in-register (no memory round trip) from two full-vector splats plus
 * one Q6_V_valign_VVR half-swap -- see the comment at the call site below for the
 * exact byte-offset derivation.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(const int8_t *Q, const int8_t *K, int8_t *S,
                      int M, int N, int D,
                      int32_t scale_mult, int scale_shift)
{
    int num_full = D >> 2;
    int tail     = D & 3;
    int num_kgroups = num_full + (tail > 0 ? 1 : 0);

    /* Kt: num_kgroups x 128 bytes. Lanes 0..15 and 16..31 both hold the same
     * K[j][k0..k0+3] (j=0..15) -- duplicated so a 2-row-packed Q vector produces
     * useful dot products in both halves. Built once, reused across all row-pairs. */
    static int8_t __attribute__((aligned(128))) Kt[34 * 128];

    for (int kg = 0; kg < num_full; kg++) {
        int8_t *blk = Kt + kg * 128;
        int k0 = kg * 4;
        for (int j = 0; j < N; j++) {
            int8_t b0 = K[j*D + k0 + 0];
            int8_t b1 = K[j*D + k0 + 1];
            int8_t b2 = K[j*D + k0 + 2];
            int8_t b3 = K[j*D + k0 + 3];
            blk[j*4+0] = b0; blk[j*4+1] = b1; blk[j*4+2] = b2; blk[j*4+3] = b3;
            blk[64 + j*4+0] = b0; blk[64 + j*4+1] = b1; blk[64 + j*4+2] = b2; blk[64 + j*4+3] = b3;
        }
        for (int j = N; j < 16; j++) {
            blk[j*4+0]=blk[j*4+1]=blk[j*4+2]=blk[j*4+3]=0;
            blk[64+j*4+0]=blk[64+j*4+1]=blk[64+j*4+2]=blk[64+j*4+3]=0;
        }
    }
    if (tail > 0) {
        int8_t *blk = Kt + num_full * 128;
        int k0 = num_full * 4;
        memset(blk, 0, 128);
        for (int j = 0; j < N; j++) {
            for (int t = 0; t < tail; t++) {
                int8_t b = K[j*D + k0 + t];
                blk[j*4+t] = b;
                blk[64 + j*4+t] = b;
            }
        }
    }

    int64_t half_val = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0LL;

    /* Precompute per-row 4-byte-packed Q words for all k-groups, once. */
    static uint32_t qwords[16][34];
    for (int i = 0; i < M; i++) {
        const int8_t *Qrow = Q + i * D;
        for (int kg = 0; kg < num_full; kg++) {
            int k0 = kg * 4;
            qwords[i][kg] = (uint32_t)((uint8_t)Qrow[k0+0])
                          | ((uint32_t)((uint8_t)Qrow[k0+1]) << 8)
                          | ((uint32_t)((uint8_t)Qrow[k0+2]) << 16)
                          | ((uint32_t)((uint8_t)Qrow[k0+3]) << 24);
        }
        if (tail > 0) {
            uint32_t q4 = 0;
            int k0 = num_full * 4;
            for (int t = 0; t < tail; t++)
                q4 |= (uint32_t)((uint8_t)Qrow[k0+t]) << (t*8);
            qwords[i][num_full] = q4;
        }
    }

    int32_t accbuf[32] __attribute__((aligned(128)));

    /* Constant selector: lanes 0..15 keep vA (row i0), lanes 16..31 keep vB (row i1). */
    for (int ip = 0; ip < M; ip += 2) {
        int i0 = ip, i1 = ip + 1;
        HVX_Vector acc = Q6_V_vzero();

        for (int kg = 0; kg < num_kgroups; kg++) {
            uint32_t w0 = qwords[i0][kg];
            uint32_t w1 = qwords[i1][kg];
            HVX_Vector vA = Q6_V_vsplat_R(w0);      /* w0 in every lane */
            HVX_Vector vB = Q6_V_vsplat_R(w1);       /* w1 in every lane */
            /* Merge in-register (no memory round trip): Q6_V_valign_VVR(Vu,Vv,Rt)
             * logically concatenates {Vu (high 128B) :: Vv (low 128B)} into 256 bytes
             * and returns the 128B window starting at byte offset Rt. With
             * Q6_V_valign_VVR(vB, vA, 64): concat = {vB hi, vA lo}; window@64 =
             * bytes[64:128) of vA followed by bytes[0:64) of vB. Since vA/vB are
             * splats (uniform 4-byte repeating pattern across all 128B), any 64B
             * half reproduces the same pattern -- so result[0:64)=vA pattern (row i0),
             * result[64:128)=vB pattern (row i1), i.e. lanes 0..15=i0, 16..31=i1. */
            HVX_Vector va = Q6_V_valign_VVR(vB, vA, 64);
            HVX_Vector vb = *(const HVX_Vector *)(Kt + kg * 128);
            acc = Q6_Vw_vrmpyacc_VwVbVb(acc, va, vb);
        }

        *(HVX_Vector *)accbuf = acc;

        int8_t *Srow0 = S + i0 * N;
        int8_t *Srow1 = S + i1 * N;
        for (int j = 0; j < N; j++) {
            int64_t r = (int64_t)accbuf[j] * (int64_t)scale_mult;
            int64_t q = (r >= 0) ? ((r + half_val) >> scale_shift)
                                  : -((-r + half_val) >> scale_shift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            Srow0[j] = (int8_t)q;
        }
        for (int j = 0; j < N; j++) {
            int64_t r = (int64_t)accbuf[16 + j] * (int64_t)scale_mult;
            int64_t q = (r >= 0) ? ((r + half_val) >> scale_shift)
                                  : -((-r + half_val) >> scale_shift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            Srow1[j] = (int8_t)q;
        }
    }
}
