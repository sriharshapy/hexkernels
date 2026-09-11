/*
 * Near-miss: off-by-one in the K-tail handling. The tail k-group (K=262 =
 * 65*4 + 2, so the last 4-byte group has only 2 valid k's: 260 and 261)
 * should zero-pad the 2 MISSING bytes (positions for the nonexistent
 * k=262,263) and keep both real tail bytes (k=260 AND k=261). This buggy
 * version loops "for (t = 0; t < tail - 1; t++)" instead of "t < tail",
 * i.e. it writes only 1 of the 2 valid tail bytes (k=260) into the packed
 * block and silently drops k=261's contribution entirely (that byte stays
 * zero-padded along with the truly-missing k=262,263) -- a very plausible
 * "pad instead of copy" tail bug. Wrong whenever A[.,261] and B[261,.] are
 * both non-zero (guaranteed by the harness's tail edge case).
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
            /* BUG: should be "t < tail" (copies all valid tail bytes);
             * "t < tail - 1" drops the LAST valid tail k (k=261) and only
             * ever zero-pads it. */
            for (int t = 0; t < tail - 1; t++)
                blk[j*4+t] = B[(k0+t)*N + j];
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
