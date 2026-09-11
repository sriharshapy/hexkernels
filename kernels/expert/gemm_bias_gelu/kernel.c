/*
 * HVX fused GEMM+bias+saturate+GELU-via-LUT for M=N=K=48.
 *
 * Strategy:
 *   - Precompute transposed-B blocks (same layout as gemm_bias_relu_requant sibling):
 *     For each k-group of 4, word lane j stores {B[k0][j], B[k1][j], B[k2][j], B[k3][j]}.
 *     This lets Q6_Vw_vrmpyacc_VwVbVb accumulate 4-wide dot product per lane.
 *   - For each output row i:
 *       * vrmpy K-reduction -> int32 accumulators
 *       * vector add bias (Q6_Vw_vadd_VwVw)
 *       * extract to scalar accbuf, then saturate to int8 and LUT-lookup (scalar epilogue)
 *         -- scalar epilogue is cheap (N=48) and ensures bit-exact sat + LUT gather.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias,
                      const int8_t *gelu_lut,
                      int8_t *out,
                      int M, int N, int K)
{
    /*
     * Transposed B layout (N=48, K=48 -> 12 k-groups):
     *   lo block (j=0..31):  128 bytes, word lane w=j -> {B[k0][j], B[k1][j], B[k2][j], B[k3][j]}
     *   hi block (j=32..47): 128 bytes, word lane w=(j-32) -> same, rest 0
     * Total: 12 groups * 256 bytes = 3072 bytes.
     */
    static int8_t __attribute__((aligned(128))) Bt[12 * 2 * 128];

    int num_kgroups = K >> 2; /* K must be multiple of 4 */

    /* Build transposed B */
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

        /* Vector bias add */
        HVX_Vector vbias_lo = *(const HVX_Vector *)(&bias[0]);   /* bias[0..31]  */
        HVX_Vector vbias_hi = *(const HVX_Vector *)(&bias[32]);  /* bias[32..47] in lanes 0..15 */

        HVX_Vector biased_lo = Q6_Vw_vadd_VwVw(acc_lo, vbias_lo);
        HVX_Vector biased_hi = Q6_Vw_vadd_VwVw(acc_hi, vbias_hi);

        /* Store biased int32s to accbuf */
        *(HVX_Vector *)(&accbuf[0])  = biased_lo;
        *(HVX_Vector *)(&accbuf[32]) = biased_hi;

        /* Scalar epilogue: saturate to int8, then LUT-lookup for GELU */
        int8_t *out_row = out + i * N;
        for (int j = 0; j < N; j++) {
            int32_t b = accbuf[j];
            /* saturate to int8 */
            if (b >  127) b =  127;
            if (b < -128) b = -128;
            int8_t pre_lut = (int8_t)b;
            /* GELU via LUT: index = (uint8_t)(pre_lut + 128) */
            uint8_t idx = (uint8_t)(pre_lut + 128);
            out_row[j] = gelu_lut[idx];
        }
    }
}
