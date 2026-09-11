/*
 * HVX fused GEMM+saturate+GELU-via-LUT for M=N=K=48 (no bias).
 *
 * Strategy:
 *   - Pretranspose B into 128-byte blocks per k-group of 4:
 *     word lane j stores {B[k0][j], B[k1][j], B[k2][j], B[k3][j]}.
 *   - For each output row i:
 *       * Q6_Vw_vrmpyacc_VwVbVb K-reduction -> int32 accumulators (lo=j0..31, hi=j32..47)
 *       * Scalar epilogue: saturate to int8, then LUT-lookup for GELU.
 *         (N=48 scalar epilogue is cheap; ensures bit-exact sat + LUT gather.)
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int8_t *gelu_lut,
                      int8_t *out,
                      int M, int N, int K)
{
    /*
     * Transposed B layout (N=48, K=48 -> 12 k-groups of 4):
     *   lo block (j=0..31):  128 bytes, word lane w=j -> {B[k0][j],B[k1][j],B[k2][j],B[k3][j]}
     *   hi block (j=32..47): 128 bytes, word lane w=(j-32) -> same, lanes 16..31 zeroed
     * Total: 12 * 256 = 3072 bytes.
     */
    static int8_t __attribute__((aligned(128))) Bt[12 * 2 * 128];

    int num_kgroups = K >> 2; /* K must be multiple of 4; 48/4 = 12 */

    /* Build transposed B once per call */
    for (int kg = 0; kg < num_kgroups; kg++) {
        int8_t * restrict blk_lo = Bt + kg * 256;
        int8_t * restrict blk_hi = Bt + kg * 256 + 128;
        const int k0 = kg * 4;
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

    /* Accumulation buffer: 64 int32s (2 HVX vectors); only [0..47] used for N=48 */
    int32_t accbuf[64] __attribute__((aligned(128)));

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;

        HVX_Vector acc_lo = Q6_V_vzero(); /* word lanes 0..31 = j=0..31 */
        HVX_Vector acc_hi = Q6_V_vzero(); /* word lanes 0..15 = j=32..47 */

        /* K-reduction via vrmpy over k-groups of 4 */
        for (int kg = 0; kg < num_kgroups; kg++) {
            const int k0 = kg * 4;
            /* Pack A[k0..k3] into an int32 word, splat across all vector lanes */
            int32_t a4 = (int32_t)((uint8_t)Arow[k0+0])
                       | (int32_t)((uint8_t)Arow[k0+1] <<  8)
                       | (int32_t)((uint8_t)Arow[k0+2] << 16)
                       | (int32_t)((uint8_t)Arow[k0+3] << 24);
            HVX_Vector va = Q6_V_vsplat_R(a4);

            HVX_Vector vb_lo = *(const HVX_Vector *)(Bt + kg * 256);
            HVX_Vector vb_hi = *(const HVX_Vector *)(Bt + kg * 256 + 128);

            acc_lo = Q6_Vw_vrmpyacc_VwVbVb(acc_lo, va, vb_lo);
            acc_hi = Q6_Vw_vrmpyacc_VwVbVb(acc_hi, va, vb_hi);
        }

        /* Store int32 accumulators to accbuf */
        *(HVX_Vector *)(&accbuf[0])  = acc_lo;
        *(HVX_Vector *)(&accbuf[32]) = acc_hi;

        /* Scalar epilogue: saturate to int8, then GELU LUT lookup */
        int8_t *out_row = out + i * N;
        for (int j = 0; j < N; j++) {
            int32_t v = accbuf[j];
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            uint8_t idx = (uint8_t)((int8_t)v + 128);
            out_row[j] = gelu_lut[idx];
        }
    }
}
