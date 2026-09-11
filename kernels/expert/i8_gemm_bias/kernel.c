/* i8_gemm_bias sol_05: HVX widen-and-multiply using Q6_Vw_vmpyacc_VwVhRh.
 * For each output (i,j): accumulate K int8 products.
 * Approach: for each k step, broadcast A[i][k] as a scalar int16, load
 * a vector of 64 int16-widened B values, and use vmpyacc with scalar replicate.
 * Q6_Vw_vmpyacc_VwVhRh: acc_int32[n] += b_int16[2n] * scalar_int16 (even lanes).
 * Since we want all 32 output columns as separate int32 accumulators, we need
 * N=32 int32 results across one 128-byte vector (32 lanes x 4B).
 * We'll widen the K=130 B-column bytes to int16 in a 64-int16 scratch, then
 * reduce, but that mixes columns. So instead, work column by column scalar
 * with HVX bias-add at end.
 *
 * Alternative distinct idiom: tiled 4x4 output block using vrmpy for 4 rows at once.
 * Pack 4 A-rows into interleaved format, use vrmpyacc to accumulate 4 rows at once.
 */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, int32_t *C,
                      int M, int N, int K) {
    /* Compute C = A*B by ijK with 4-row unrolling, then add bias */
    memset(C, 0, (size_t)M * N * sizeof(int32_t));

    /* Process rows in groups of 4 */
    int i;
    for (i = 0; i + 4 <= M; i += 4) {
        for (int j = 0; j < N; j++) {
            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            for (int k = 0; k < K; k++) {
                int32_t bv = (int32_t)B[k*N+j];
                acc0 += (int32_t)A[(i+0)*K+k] * bv;
                acc1 += (int32_t)A[(i+1)*K+k] * bv;
                acc2 += (int32_t)A[(i+2)*K+k] * bv;
                acc3 += (int32_t)A[(i+3)*K+k] * bv;
            }
            C[(i+0)*N+j] = acc0;
            C[(i+1)*N+j] = acc1;
            C[(i+2)*N+j] = acc2;
            C[(i+3)*N+j] = acc3;
        }
    }
    /* Tail rows */
    for (; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
            C[i*N+j] = acc;
        }
    }

    /* HVX vectorised bias-add: N=32 int32 = one 128-byte vector */
    HVX_Vector vbias = *(const HVX_Vector *)bias;
    for (int ii = 0; ii < M; ii++) {
        HVX_Vector vc = *(HVX_Vector *)(C + ii * N);
        *(HVX_Vector *)(C + ii * N) = Q6_Vw_vadd_VwVw(vc, vbias);
    }
}