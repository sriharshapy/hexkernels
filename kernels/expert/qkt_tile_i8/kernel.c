/*
 * HVX QK^T tile (column-major K), row-pair-packed vrmpy broadcast.
 * M=16, N=16, D=98.
 *
 * K is stored column-major-per-key: K[d*N+j]. For a fixed reduction
 * k-group (d = kg*4 .. kg*4+3), the 4 rows K[k0*N..], K[(k0+1)*N..], ...
 * are each CONTIGUOUS over j -- so building the interleaved Kt block
 * ("4 K-bytes per key, consecutive keys") is a plain gather over 4 rows,
 * no true transpose needed.
 *
 * Packing (identical shape to the row-major-K sibling task's expert):
 *   Kt[kg*128 + j*4 + r]      = K[(kg*4+r)*N + j]   for j in [0,16)  (lanes 0..15)
 *   Kt[kg*128 + 64 + j*4 + r] = K[(kg*4+r)*N + j]   for j in [0,16)  (lanes 16..31, dup)
 *
 * Per row-pair (i0,i1), per k-group kg:
 *   va lane j<16  = Q[i0][k0..k0+3]  (splat)
 *   va lane j>=16 = Q[i1][k0..k0+3]  (splat)
 *   vrmpyacc(acc, va, vb) -> acc[j<16] += Q[i0].K[:,j], acc[j>=16] += Q[i1].K[:,j]
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
     * K[:,j] (j=0..15) 4-byte group for reduction group kg -- duplicated so
     * a 2-row-packed Q vector produces useful dot products in both halves.
     * D=98 -> num_kgroups = 25; size for the pinned max (32 groups covers
     * D up to 128). */
    static int8_t __attribute__((aligned(128))) Kt[32 * 128];

    for (int kg = 0; kg < num_full; kg++) {
        int8_t *blk = Kt + kg * 128;
        int k0 = kg * 4;
        for (int j = 0; j < N; j++) {
            int8_t b0 = K[(k0+0)*N + j];
            int8_t b1 = K[(k0+1)*N + j];
            int8_t b2 = K[(k0+2)*N + j];
            int8_t b3 = K[(k0+3)*N + j];
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
                int8_t b = K[(k0+t)*N + j];
                blk[j*4+t] = b;
                blk[64 + j*4+t] = b;
            }
        }
    }

    int64_t half_val = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0LL;

    /* Precompute per-row 4-byte-packed Q words for all k-groups, once. */
    static uint32_t qwords[16][32];
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

    for (int ip = 0; ip < M; ip += 2) {
        int i0 = ip, i1 = ip + 1;
        HVX_Vector acc = Q6_V_vzero();

        for (int kg = 0; kg < num_kgroups; kg++) {
            uint32_t w0 = qwords[i0][kg];
            uint32_t w1 = qwords[i1][kg];
            HVX_Vector vA = Q6_V_vsplat_R(w0);      /* w0 in every lane */
            HVX_Vector vB = Q6_V_vsplat_R(w1);       /* w1 in every lane */
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
