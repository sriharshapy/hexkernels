/*
 * gemm_tile_deepk_i8 Solution 1: full B pretranspose into 128B k-group
 * blocks, chained Q6_Vw_vrmpyacc_VwVbVb reduction (same technique as
 * i8_gemm_gelu_lut's Bt packing, generalized to the deep K=262 tail).
 *
 * C[i*N+j] = sum_k A[i*K+k]*B[k*N+j]. B is [K x N] CONVENTIONAL layout.
 * M=8, N=8, K=262 -> 65 full k-groups of 4 + 1 tail k-group (2 valid k's,
 * zero-padded to 4).
 *
 * Packing: Bt[kg*128 + j*4 + r] = B[(kg*4+r)*N + j] for j in [0,N), zero
 * elsewhere (only the low 8 word-lanes of each 128B block hold real data
 * since N=8; the rest are zero and never read back).
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C,
                      int M, int N, int K)
{
    int num_full = K >> 2;
    int tail     = K & 3;
    int num_kgroups = num_full + (tail > 0 ? 1 : 0);

    /* Room for K up to 264 (66 groups); pinned K=262 -> 66 groups exactly. */
    static int8_t __attribute__((aligned(128))) Bt[70 * 128];

    for (int kg = 0; kg < num_full; kg++) {
        int8_t *blk = Bt + kg * 128;
        int k0 = kg * 4;
        memset(blk, 0, 128);
        for (int j = 0; j < N; j++) {
            blk[j*4+0] = B[(k0+0)*N + j];
            blk[j*4+1] = B[(k0+1)*N + j];
            blk[j*4+2] = B[(k0+2)*N + j];
            blk[j*4+3] = B[(k0+3)*N + j];
        }
    }
    if (tail > 0) {
        int8_t *blk = Bt + num_full * 128;
        int k0 = num_full * 4;
        memset(blk, 0, 128);
        for (int j = 0; j < N; j++) {
            for (int t = 0; t < tail; t++)
                blk[j*4+t] = B[(k0+t)*N + j];
            /* remaining (4-tail) bytes of this lane stay zero-padded */
        }
    }

    int32_t accbuf[32] __attribute__((aligned(128)));

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;
        HVX_Vector acc = Q6_V_vzero();

        for (int kg = 0; kg < num_kgroups; kg++) {
            int k0 = kg * 4;
            uint32_t a0 = (k0+0 < K) ? (uint8_t)Arow[k0+0] : 0;
            uint32_t a1 = (k0+1 < K) ? (uint8_t)Arow[k0+1] : 0;
            uint32_t a2 = (k0+2 < K) ? (uint8_t)Arow[k0+2] : 0;
            uint32_t a3 = (k0+3 < K) ? (uint8_t)Arow[k0+3] : 0;
            uint32_t w = a0 | (a1 << 8) | (a2 << 16) | (a3 << 24);
            HVX_Vector va = Q6_V_vsplat_R(w);
            HVX_Vector vb = *(const HVX_Vector *)(Bt + kg * 128);
            acc = Q6_Vw_vrmpyacc_VwVbVb(acc, va, vb);
        }

        *(HVX_Vector *)accbuf = acc;
        memcpy(C + i * N, accbuf, (size_t)N * sizeof(int32_t));
    }
}
