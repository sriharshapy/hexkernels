/*
 * matmul_residual_add_i8 Solution 2: structurally different from s1 -- no
 * k-group chaining at all. Since K=90 <= 128, each B column j is gathered
 * ONCE (strided by N) into a contiguous zero-padded 128B buffer (reused
 * across all M rows); each row's A is gathered into a zero-padded 128B
 * buffer; a SINGLE Q6_Vw_vrmpyacc_VwVbVb + full 32-lane horizontal reduce
 * gives the whole-K dot product per (i,j) directly.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>

static inline int32_t hreduce32(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

void candidate_kernel(const int8_t *A, const int8_t *B, const int8_t *residual,
                      int8_t *out, int M, int N, int K,
                      int32_t scale_mult, int scale_shift)
{
    static int8_t vcols[16][128] __attribute__((aligned(128)));
    static int8_t vrow[128] __attribute__((aligned(128)));

    /* Gather each B column (strided by N) into a contiguous, zero-padded
     * 128B buffer -- built ONCE, reused across all M rows. */
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) vcols[j][k] = B[k*N + j];
        for (int k = K; k < 128; k++) vcols[j][k] = 0;
    }

    int64_t half_val = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0LL;
    const HVX_Vector zero = Q6_V_vzero();

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;
        for (int k = 0; k < K; k++) vrow[k] = Arow[k];
        for (int k = K; k < 128; k++) vrow[k] = 0;
        HVX_Vector va = *(const HVX_Vector *)vrow;

        const int8_t *resrow = residual + i * N;
        int8_t *outrow = out + i * N;
        for (int j = 0; j < N; j++) {
            HVX_Vector vb = *(const HVX_Vector *)vcols[j];
            int32_t rawacc = hreduce32(Q6_Vw_vrmpyacc_VwVbVb(zero, va, vb));

            int64_t r = (int64_t)rawacc * (int64_t)scale_mult;
            int64_t q = (r >= 0) ? ((r + half_val) >> scale_shift)
                                  : -((-r + half_val) >> scale_shift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            int32_t requantized = (int32_t)q;

            int32_t sum = requantized + (int32_t)resrow[j];
            if (sum >  127) sum =  127;
            if (sum < -128) sum = -128;
            outrow[j] = (int8_t)sum;
        }
    }
}
