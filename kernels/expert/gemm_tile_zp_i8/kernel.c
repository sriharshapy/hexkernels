/*
 * gemm_tile_zp_i8 Solution 2: structurally different from s1 in BOTH the
 * colsum method AND the rawdot reduction shape.
 *
 *   - colsum[j]: a PLAIN scalar integer add-reduction directly over B's
 *     column (no vrmpy at all) -- the "structurally different reduction"
 *     alternative.
 *   - rawdot[i,j]: since K=94 <= 128, each column j is gathered ONCE
 *     (strided by N in B) into a contiguous zero-padded 128B buffer (reused
 *     across all M rows), and each row's A is gathered into a zero-padded
 *     128B buffer; a SINGLE Q6_Vw_vrmpyacc_VwVbVb + full 32-lane horizontal
 *     reduce gives the whole-K dot product per (i,j) -- no k-group chaining
 *     loop at all (contrast with s1's chained-accumulate-over-groups).
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

void candidate_kernel(const int8_t *A, const int8_t *B, int8_t *out,
                      int M, int N, int K,
                      int32_t zpA, int32_t scale_mult, int scale_shift)
{
    static int8_t vcols[16][128] __attribute__((aligned(128)));
    static int8_t vrow[128] __attribute__((aligned(128)));
    int32_t colsum[16];

    /* Gather each B column (strided by N) into a contiguous, zero-padded
     * 128B buffer, and independently compute its plain-scalar colsum. */
    for (int j = 0; j < N; j++) {
        int32_t cs = 0;
        for (int k = 0; k < K; k++) {
            int8_t b = B[k*N + j];
            vcols[j][k] = b;
            cs += (int32_t)b;
        }
        for (int k = K; k < 128; k++) vcols[j][k] = 0;
        colsum[j] = cs;
    }

    int64_t half_val = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0LL;
    const HVX_Vector zero = Q6_V_vzero();

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;
        for (int k = 0; k < K; k++) vrow[k] = Arow[k];
        for (int k = K; k < 128; k++) vrow[k] = 0;
        HVX_Vector va = *(const HVX_Vector *)vrow;

        int8_t *outrow = out + i * N;
        for (int j = 0; j < N; j++) {
            HVX_Vector vb = *(const HVX_Vector *)vcols[j];
            int32_t rawdot = hreduce32(Q6_Vw_vrmpyacc_VwVbVb(zero, va, vb));
            int32_t accv = rawdot - zpA * colsum[j];
            int64_t r = (int64_t)accv * (int64_t)scale_mult;
            int64_t q = (r >= 0) ? ((r + half_val) >> scale_shift)
                                  : -((-r + half_val) >> scale_shift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            outrow[j] = (int8_t)q;
        }
    }
}
