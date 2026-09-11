/*
 * Attention A·V tile HVX kernel.
 * O[i*D+d] = sum_{j=0}^{N-1} A[i*N+j] * V[j*D+d]   (int32 accumulate)
 * Pinned: M=16, N=16, D=130.
 *
 * Strategy: pretranspose V into Vt so that for each k-group of 4 (kg=0..3)
 * and each d in 0..127, the bytes {V[j0*D+d], V[j1*D+d], V[j2*D+d], V[j3*D+d]}
 * (j0=kg*4, j1..j3 = kg*4+1..3) are packed consecutively.
 * Layout: Vt[kg * 512 + d * 4 + q] = V[(kg*4+q)*D+d], d=0..127, q=0..3.
 * Each HVX_Vector (128B = 32 word-lanes) covers 32 d-values.
 * 4 HVX_Vectors per kg-group cover d=0..127.
 *
 * Inner loop per row i:
 *   acc_w[kg][d_blk] = Q6_Vw_vrmpyacc(acc, splat(A[i][kg*4..+3]), Vt block)
 * Then scatter int32 accumulators to O[i*D+d].
 * Tail d=128..129 handled with scalar dot product.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

#define _M  16
#define _N  16
#define _D  130

/* N/4 = 4 k-groups, D_VEC = 128 (multiple of 32), TAIL = D-D_VEC = 2 */
#define KG       4       /* N/4 */
#define D_VEC  128       /* vectorized d extent */
#define D_BLKS   4       /* D_VEC/32 */

void candidate_kernel(const int8_t *A, const int8_t *V, int32_t *O,
                      int M, int N, int D)
{
    /*
     * Vt: transposed and repacked V for vrmpy.
     * KG groups x D_VEC d-values x 4 j-bytes = 4*128*4 = 2048 bytes.
     * Plus tail scalars handled separately (no buffer needed).
     */
    static int8_t __attribute__((aligned(128))) Vt[KG * D_VEC * 4];

    /* Build Vt: Vt[kg*D_VEC*4 + d*4 + q] = V[(kg*4+q)*D + d] */
    for (int kg = 0; kg < KG; kg++) {
        int8_t *base = Vt + kg * D_VEC * 4;
        for (int d = 0; d < D_VEC; d++) {
            base[d*4+0] = V[(kg*4+0)*D + d];
            base[d*4+1] = V[(kg*4+1)*D + d];
            base[d*4+2] = V[(kg*4+2)*D + d];
            base[d*4+3] = V[(kg*4+3)*D + d];
        }
    }

    /*
     * Per output row i: accumulate into 4 HVX_Vectors (acc0..acc3),
     * each covering 32 word-lanes = 32 d-values.
     * Then store to O[i*D + d_block*32 .. +31] as int32.
     * Finally handle tail d=128..129 in scalar.
     */

    /* Aligned staging buffer for O's 128 int32 vectorized values */
    int32_t __attribute__((aligned(128))) obuf[D_VEC]; /* 128 ints = 512 bytes */

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * N;

        /* Initialize four 32-wide word accumulators */
        HVX_Vector acc0 = Q6_V_vzero();
        HVX_Vector acc1 = Q6_V_vzero();
        HVX_Vector acc2 = Q6_V_vzero();
        HVX_Vector acc3 = Q6_V_vzero();

        for (int kg = 0; kg < KG; kg++) {
            /* Pack A[i][kg*4..kg*4+3] as signed bytes into word, splat */
            int k0 = kg * 4;
            /* vrmpy treats bytes as signed; pack in byte lanes 0,1,2,3 of word */
            int32_t a4 = ((uint32_t)(uint8_t)Arow[k0+0])
                       | ((uint32_t)(uint8_t)Arow[k0+1] <<  8)
                       | ((uint32_t)(uint8_t)Arow[k0+2] << 16)
                       | ((uint32_t)(uint8_t)Arow[k0+3] << 24);
            HVX_Vector va = Q6_V_vsplat_R(a4);

            const int8_t *vbase = Vt + kg * D_VEC * 4;
            /* 4 blocks of 32 d-values = 4 x 128-byte vectors */
            HVX_Vector vb0 = *(const HVX_Vector *)(vbase + 0*128);
            HVX_Vector vb1 = *(const HVX_Vector *)(vbase + 1*128);
            HVX_Vector vb2 = *(const HVX_Vector *)(vbase + 2*128);
            HVX_Vector vb3 = *(const HVX_Vector *)(vbase + 3*128);

            acc0 = Q6_Vw_vrmpyacc_VwVbVb(acc0, va, vb0);
            acc1 = Q6_Vw_vrmpyacc_VwVbVb(acc1, va, vb1);
            acc2 = Q6_Vw_vrmpyacc_VwVbVb(acc2, va, vb2);
            acc3 = Q6_Vw_vrmpyacc_VwVbVb(acc3, va, vb3);
        }

        /* Store vectorized accumulators */
        *(HVX_Vector *)(obuf +  0) = acc0;
        *(HVX_Vector *)(obuf + 32) = acc1;
        *(HVX_Vector *)(obuf + 64) = acc2;
        *(HVX_Vector *)(obuf + 96) = acc3;

        /* Copy vectorized part to output */
        int32_t *Orow = O + i * D;
        memcpy(Orow, obuf, D_VEC * sizeof(int32_t));

        /* Tail: d = 128..129 (scalar) */
        for (int d = D_VEC; d < D; d++) {
            int32_t acc = 0;
            for (int j = 0; j < N; j++)
                acc += (int32_t)Arow[j] * (int32_t)V[j*D + d];
            Orow[d] = acc;
        }
    }
}
