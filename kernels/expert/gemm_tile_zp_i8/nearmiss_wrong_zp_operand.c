/* Near-miss: applies the zero-point correction using a sum over the WRONG
 * operand/axis. The correct decomposition is
 *   acc[i,j] = rawdot[i,j] - zpA * colsum[j]     (colsum[j] = sum_k B[k,j])
 * This buggy version instead computes
 *   acc[i,j] = rawdot[i,j] - zpA * rowsum_of_A[i]  (rowsum_of_A[i] = sum_k A[i,k])
 * i.e. it subtracts zpA times a sum over A's own row (varying with i, not
 * j) instead of zpA times B's column sum (varying with j) -- a very
 * plausible "picked the wrong operand's reduction" bug. Compiles fine;
 * guaranteed wrong whenever zpA != 0 (the harness sweeps a negative and a
 * positive zpA, so this is always caught -- the zpA=0 set alone would NOT
 * catch it, which is exactly why the harness also sweeps zpA != 0).
 */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B, int8_t *out,
                      int M, int N, int K,
                      int32_t zpA, int32_t scale_mult, int scale_shift)
{
    for (int i = 0; i < M; i++) {
        int32_t rowsum_of_A = 0;
        for (int k = 0; k < K; k++) rowsum_of_A += (int32_t)A[i*K + k];

        for (int j = 0; j < N; j++) {
            int32_t rawdot = 0;
            for (int k = 0; k < K; k++)
                rawdot += (int32_t)A[i*K + k] * (int32_t)B[k*N + j];

            /* BUG: should subtract zpA*colsum[j] (sum over B's column j),
             * not zpA*rowsum_of_A (sum over A's row i). */
            int32_t acc = rawdot - zpA * rowsum_of_A;

            int64_t r    = (int64_t)acc * (int64_t)scale_mult;
            int64_t half = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0;
            int64_t q    = (r >= 0) ? ((r + half) >> scale_shift)
                                    : -((-r + half) >> scale_shift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            out[i*N + j] = (int8_t)q;
        }
    }
}
