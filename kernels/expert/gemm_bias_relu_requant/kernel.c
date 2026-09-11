/*
 * HVX fused GEMM+bias+ReLU+requant for M=N=K=48.
 *
 * Strategy:
 *   - Precompute transposed B blocks so that for each k-group of 4, all N=48 columns
 *     are accessed with a consistent 4-byte-per-word-lane layout.
 *   - For each output row i:
 *     - Use Q6_Vw_vrmpyacc_VwVbVb to accumulate: splat A[i][k:k+3] vs Bt block.
 *       For word lane j: acc[j] += A[k]*B[k][j] + A[k+1]*B[k+1][j] + A[k+2]*B[k+2][j] + A[k+3]*B[k+3][j]
 *     - After full K reduction: add bias (vector), ReLU (vector vmax), then scalar requant.
 *
 * Requant is done in scalar (int64) to match reference exactly.
 * The GEMM accumulation is fully vectorized using vrmpy.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, int8_t *out,
                      int M, int N, int K,
                      int32_t mult, int shift, int8_t zp)
{
    /*
     * Transposed B:
     *   For each k-group kg (K/4 groups) and each output column j (0..N-1):
     *   Bt[kg*N*4 + j*4 + q] = B[(kg*4+q)*N+j] for q=0..3
     *
     * With N=48, K=48: 12 groups × 48 cols × 4 bytes = 2304 bytes.
     * We split into two aligned 128-byte blocks per k-group:
     *   lo block: j=0..31 (128 bytes, word lane w=j has bytes B[k0..k0+3][j])
     *   hi block: j=32..47 (64 valid bytes, word lanes 0..15 used, rest 0)
     *
     * Each block is 128-byte aligned for direct HVX vector load.
     */
    static int8_t __attribute__((aligned(128))) Bt[12 * 2 * 128]; /* 12 kgroups × 2 × 128 */

    int num_kgroups = K >> 2; /* assume K multiple of 4 */

    /* Build transposed B */
    for (int kg = 0; kg < num_kgroups; kg++) {
        int8_t * restrict blk_lo = Bt + kg * 256;
        int8_t * restrict blk_hi = Bt + kg * 256 + 128;
        const int k0 = kg * 4;
        /* clear hi block (unused lanes beyond N=48, i.e. j=48..63) */
        memset(blk_hi, 0, 128);
        for (int j = 0; j < N && j < 32; j++) {
            blk_lo[j*4+0] = B[(k0+0)*N+j];
            blk_lo[j*4+1] = B[(k0+1)*N+j];
            blk_lo[j*4+2] = B[(k0+2)*N+j];
            blk_lo[j*4+3] = B[(k0+3)*N+j];
        }
        for (int j = 32; j < N; j++) {
            int jj = j - 32;
            blk_hi[jj*4+0] = B[(k0+0)*N+j];
            blk_hi[jj*4+1] = B[(k0+1)*N+j];
            blk_hi[jj*4+2] = B[(k0+2)*N+j];
            blk_hi[jj*4+3] = B[(k0+3)*N+j];
        }
    }

    /* Precompute requant constants */
    int64_t half_val = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int32_t izp = (int32_t)zp;

    /* Accumulation buffer: N=48 int32 values, aligned for HVX extraction */
    int32_t accbuf[64] __attribute__((aligned(128))); /* 64 ints = 256 bytes, 2 HVX vectors */

    HVX_Vector vzero = Q6_V_vzero();

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;

        HVX_Vector acc_lo = Q6_V_vzero(); /* word lanes 0..31 = j=0..31 */
        HVX_Vector acc_hi = Q6_V_vzero(); /* word lanes 0..15 = j=32..47 */

        /* K-reduction via vrmpy over k-groups of 4 */
        for (int kg = 0; kg < num_kgroups; kg++) {
            /* Pack A[i][k0..k0+3] into int32 word, splat across all vector lanes.
             * Low byte = A[k0], next = A[k0+1], ..., high = A[k0+3].
             * vrmpy reads each byte signed, so the bit pattern is what matters. */
            const int k0 = kg * 4;
            int32_t a4 = (int32_t)((uint8_t)Arow[k0+0])
                       | (int32_t)((uint8_t)Arow[k0+1] << 8)
                       | (int32_t)((uint8_t)Arow[k0+2] << 16)
                       | (int32_t)((uint8_t)Arow[k0+3] << 24);
            HVX_Vector va = Q6_V_vsplat_R(a4);

            HVX_Vector vb_lo = *(const HVX_Vector *)(Bt + kg * 256);
            HVX_Vector vb_hi = *(const HVX_Vector *)(Bt + kg * 256 + 128);

            /* acc[w] += A[k0]*Bt[j*4+0] + A[k0+1]*Bt[j*4+1] + ... (w=j for lo, w=(j-32) for hi) */
            acc_lo = Q6_Vw_vrmpyacc_VwVbVb(acc_lo, va, vb_lo);
            acc_hi = Q6_Vw_vrmpyacc_VwVbVb(acc_hi, va, vb_hi);
        }

        /* Load bias into vectors and add */
        HVX_Vector vbias_lo = *(const HVX_Vector *)(&bias[0]);   /* bias[0..31] */
        HVX_Vector vbias_hi = *(const HVX_Vector *)(&bias[32]);  /* bias[32..47] in lanes 0..15 */

        HVX_Vector biased_lo = Q6_Vw_vadd_VwVw(acc_lo, vbias_lo);
        HVX_Vector biased_hi = Q6_Vw_vadd_VwVw(acc_hi, vbias_hi);

        /* ReLU: max(biased, 0) */
        HVX_Vector relu_lo = Q6_Vw_vmax_VwVw(biased_lo, vzero);
        HVX_Vector relu_hi = Q6_Vw_vmax_VwVw(biased_hi, vzero);

        /* Store to accbuf for scalar requant */
        *(HVX_Vector *)(&accbuf[0])  = relu_lo;
        *(HVX_Vector *)(&accbuf[32]) = relu_hi;

        /* Scalar requant per output column: int64 arithmetic for exact match */
        int8_t *out_row = out + i * N;
        for (int j = 0; j < N; j++) {
            int32_t rv = accbuf[j];  /* after relu, rv >= 0 */
            int64_t v = (int64_t)rv * (int64_t)mult;
            int64_t r;
            if (v >= 0) r = (v + half_val) >> shift;
            else        r = -((-v + half_val) >> shift);
            r += izp;
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            out_row[j] = (int8_t)r;
        }
    }
}
