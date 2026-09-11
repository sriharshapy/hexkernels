/*
 * HVX fused GEMM+requant for M=32, N=32, K=130.
 *
 * Strategy:
 *   - Pretranspose B so for each k-group of 4, word lane j holds
 *     {B[k0][j], B[k0+1][j], B[k0+2][j], B[k0+3][j]}.
 *   - N=32 fits in a single HVX vector (32 int32 word lanes = 128 bytes).
 *   - K=130 = 32 full k-groups + 2 tail bytes; handle tail with scalar.
 *   - Scalar requant per output element (int64) for bit-exact rounding.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(const int8_t *A, const int8_t *B, int8_t *out,
                      int M, int N, int K,
                      int32_t mult, int shift, int8_t zp)
{
    /* K=130, N=32 are pinned by harness — specialize for speed */
    /* K groups: 32 full groups of 4 + 2 tail elements */
    const int num_kgroups = K >> 2;          /* 32 full groups */
    const int k_tail_start = num_kgroups * 4; /* 128 */

    /*
     * Transposed B layout: for each k-group kg (0..31):
     *   Bt[kg * 128 + j*4 + q] = B[(kg*4+q)*N + j]  for j=0..31, q=0..3
     * Each block is exactly 128 bytes = one HVX vector.
     */
    static int8_t __attribute__((aligned(128))) Bt[32 * 128];

    for (int kg = 0; kg < num_kgroups; kg++) {
        int8_t * restrict blk = Bt + kg * 128;
        const int k0 = kg * 4;
        for (int j = 0; j < 32; j++) {
            blk[j*4+0] = B[(k0+0)*N+j];
            blk[j*4+1] = B[(k0+1)*N+j];
            blk[j*4+2] = B[(k0+2)*N+j];
            blk[j*4+3] = B[(k0+3)*N+j];
        }
    }

    /* Requant constants */
    int64_t half_val = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int32_t izp = (int32_t)zp;

    /* Aligned accumulation buffer: 32 int32 = 128 bytes = one HVX vector */
    int32_t __attribute__((aligned(128))) accbuf[32];

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;

        /* HVX accumulator: 32 word lanes = 32 output columns */
        HVX_Vector vacc = Q6_V_vzero();

        /* K-reduction: 32 full groups of 4 */
        for (int kg = 0; kg < num_kgroups; kg++) {
            const int k0 = kg * 4;
            /* Pack A[i][k0..k0+3] into int32, splat across all 32 word lanes */
            int32_t a4 = (int32_t)((uint8_t)Arow[k0+0])
                       | ((int32_t)((uint8_t)Arow[k0+1]) <<  8)
                       | ((int32_t)((uint8_t)Arow[k0+2]) << 16)
                       | ((int32_t)((uint8_t)Arow[k0+3]) << 24);
            HVX_Vector va = Q6_V_vsplat_R(a4);
            HVX_Vector vb = *(const HVX_Vector *)(Bt + kg * 128);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, va, vb);
        }

        /* Store accumulator to scalar buffer */
        *(HVX_Vector *)accbuf = vacc;

        /* Tail: K=130, k=128..129 (2 elements), pure scalar add-into-accbuf */
        for (int j = 0; j < N; j++) {
            int32_t tail_acc = accbuf[j];
            for (int k = k_tail_start; k < K; k++) {
                tail_acc += (int32_t)Arow[k] * (int32_t)B[k*N+j];
            }
            accbuf[j] = tail_acc;
        }

        /* Scalar requant per output column — int64 for bit-exact rounding */
        int8_t *out_row = out + i * N;
        for (int j = 0; j < N; j++) {
            int64_t v = (int64_t)accbuf[j] * (int64_t)mult;
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
