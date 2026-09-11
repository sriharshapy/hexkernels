/* Near-miss: packs X as if it were [N x K] row-major (like gemm_tile_i8's
 * transposed B) instead of the CORRECT [K x N] row-major layout this task
 * actually uses. A very plausible row/column-major mixup bug: BT_pack[k4*128
 * + j*4+r] = X[j*K + (k4*4+r)] instead of the correct X[(k4*4+r)*N + j].
 * Compiles fine (index stays in-bounds for these shapes) but produces
 * systematically wrong dot products. */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define MAX_NGROUPS 2026

void candidate_kernel(const int8_t *A, const int8_t *X, int32_t *C,
                      int M, int K, int N) {
    int Kpad = (K + 3) & ~3;
    int ngroups = Kpad / 4;

    static int8_t __attribute__((aligned(128))) BT_pack[MAX_NGROUPS * 128];
    memset(BT_pack, 0, (size_t)ngroups * 128);
    for (int k = 0; k < K; k++) {
        int k4 = k / 4, r = k % 4;
        int8_t *dst = BT_pack + k4 * 128 + r;
        for (int j = 0; j < N; j++)
            dst[j * 4] = X[(size_t)j * K + k];   /* BUG: treats X as [N x K] */
    }

    for (int m = 0; m < M; m++) {
        const int8_t *Arow = A + (size_t)m * K;
        HVX_Vector vacc = Q6_V_vzero();

        for (int k4 = 0; k4 < ngroups; k4++) {
            int kb = k4 * 4;
            uint32_t a0 = (kb + 0 < K) ? (uint8_t)Arow[kb + 0] : 0;
            uint32_t a1 = (kb + 1 < K) ? (uint8_t)Arow[kb + 1] : 0;
            uint32_t a2 = (kb + 2 < K) ? (uint8_t)Arow[kb + 2] : 0;
            uint32_t a3 = (kb + 3 < K) ? (uint8_t)Arow[kb + 3] : 0;
            uint32_t w = a0 | (a1 << 8) | (a2 << 16) | (a3 << 24);
            HVX_Vector va = Q6_V_vsplat_R(w);
            HVX_Vector vb = *(const HVX_Vector *)(BT_pack + k4 * 128);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, va, vb);
        }

        int32_t tmp[32] __attribute__((aligned(128)));
        *(HVX_Vector *)tmp = vacc;
        memcpy(C + (size_t)m * N, tmp, (size_t)N * sizeof(int32_t));
    }
}
