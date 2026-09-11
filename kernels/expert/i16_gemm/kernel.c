/* i16_gemm sol_04: HVX Q6_Vw_vdmpy_VhRh_sat with scalar broadcast.
 * Q6_Vw_vdmpyacc_VwVhRh_sat: acc[n] += sat32(v[2n]*Rh + v[2n+1]*Rh) = sat32(Rh*(v[2n]+v[2n+1]))
 * Wait - this is a 2-deep scalar*int16 dot: acc[n] += a[2n]*scalar + a[2n+1]*scalar
 * That collapses adjacent pairs, not useful for matmul.
 *
 * Better: use Q6_Vw_vmpyiacc_VwVwRh: acc[n] += v[n] * scalar_int16
 * where v is int32 and scalar is int16 replicated. This adds a scalar-multiplied
 * int32 vector. Not what we want either.
 *
 * Use vdmpy_VhVh_sat to do ALL N=32 output columns at once for a given k-pair:
 * Pack B[k][0..31] and B[k+1][0..31] interleaved into one 64-int16 vector,
 * A[i][k] and A[i][k+1] similarly. Then vdmpy gives 32 int32 partial sums.
 *
 * Interleaved layout: va = [A[k], A[k+1], A[k], A[k+1], ...] (32 pairs)
 *                     vb = [B[k][j], B[k+1][j]] for j=0..31
 * But then each lane gets A[k]*B[k][j] + A[k+1]*B[k+1][j] -- that IS the 2-step K reduction!
 * So we can do all 32 output columns with one vdmpy call per k-pair.
 */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(const int16_t *A, const int16_t *B, int32_t *C,
                      int M, int N, int K) {
    /* N=32, M=32, K=130 */
    /* Interleave consecutive B rows in pairs: Bpair[k/2][j] = {B[k][j], B[k+1][j]} */
    int Kpad = (K + 1) & ~1; /* round to even */
    static int16_t Bpair[256 * 32]; /* Kpad/2 pairs * N=32 elements * 2 int16 each = Kpad*N */
    memset(Bpair, 0, sizeof(Bpair));
    for (int k = 0; k < K; k += 2) {
        for (int j = 0; j < N; j++) {
            Bpair[(k/2)*N*2 + j*2 + 0] = B[(k+0)*N + j];
            Bpair[(k/2)*N*2 + j*2 + 1] = (k+1 < K) ? B[(k+1)*N + j] : 0;
        }
    }
    /* Bpair[(k/2)][j] stored as interleaved pairs: index = (k/2)*64 + j*2 + 0/1 */
    /* So for each k/2 step, we have N*2=64 int16 values per group */

    for (int i = 0; i < M; i++) {
        const int16_t *Arow = A + i * K;
        int32_t       *Crow = C + i * N;
        HVX_Vector vacc = Q6_V_vzero();

        for (int k = 0; k < Kpad; k += 2) {
            int16_t a0 = Arow[k], a1 = (k+1 < K) ? Arow[k+1] : 0;
            /* Build va: replicate {a0, a1} across 32 lanes */
            /* Use Q6_Vw_vdmpy_VhVh_sat: va[2n]=a0, va[2n+1]=a1 for all n */
            /* Easiest: fill a 128-byte buffer with N=32 copies of {a0,a1} */
            int16_t va_buf[64];
            for (int t = 0; t < 32; t++) { va_buf[t*2+0] = a0; va_buf[t*2+1] = a1; }
            HVX_Vector va = *(const HVX_Vector *)va_buf;
            HVX_Vector vb = *(const HVX_Vector *)(Bpair + (k/2)*64);
            vacc = Q6_Vw_vdmpyacc_VwVhVh_sat(vacc, va, vb);
        }
        /* Store 32 int32 directly */
        *(HVX_Vector *)Crow = vacc;
    }
}