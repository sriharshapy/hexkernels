/*
 * gemm_tile_bias_relu_i8 Solution 1: tile-vectorized k4-group vrmpy accumulation
 * (same B-transpose trick as gemm_tile_i8) + vectorized bias-add/relu/clamp
 * epilogue, narrowed to int8 via a scalar copy-out (only N=20 of 32 lanes are
 * real output).
 */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *A, const int8_t *B, const int32_t *bias, int8_t *C,
                      int M, int N, int K) {
    int Kpad = (K + 3) & ~3;
    int ngroups = Kpad / 4;

    static int8_t __attribute__((aligned(128))) BT_pack[32 * 128];
    memset(BT_pack, 0, (size_t)ngroups * 128);
    for (int k = 0; k < K; k++) {
        int k4 = k / 4, r = k % 4;
        int8_t *dst = BT_pack + k4 * 128 + r;
        for (int j = 0; j < N; j++)
            dst[j * 4] = B[j * K + k];
    }

    int32_t biasbuf[32] __attribute__((aligned(128))) = {0};
    for (int j = 0; j < N; j++) biasbuf[j] = bias[j];
    HVX_Vector vbias = *(HVX_Vector *)biasbuf;
    HVX_Vector vzero = Q6_V_vzero();
    HVX_Vector vhi   = Q6_V_vsplat_R((uint32_t)127);

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;
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

        HVX_Vector vsum   = Q6_Vw_vadd_VwVw(vacc, vbias);     /* bias BEFORE relu */
        HVX_Vector vrelu  = Q6_Vw_vmax_VwVw(vsum, vzero);
        HVX_Vector vclamp = Q6_Vw_vmin_VwVw(vrelu, vhi);

        int32_t tmp[32] __attribute__((aligned(128)));
        *(HVX_Vector *)tmp = vclamp;
        int8_t *Crow = C + i * N;
        for (int j = 0; j < N; j++) Crow[j] = (int8_t)tmp[j];
    }
}
