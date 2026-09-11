/* EXPERT (fastest measured solution -- see solutions/s1.c, identical). *
 * HVX fused GEMM(+bias)+saturate+GELU-via-LUT, M=8 N=8 K=90.
 *
 * B is already TRANSPOSED ([N x K], row j contiguous over K), so building the
 * per-k-group interleaved block is a plain 4-byte gather per row (no true
 * transpose needed, unlike the [K x N] sibling tasks):
 *   Bt[kg*128 + j*4 + r] = B[j*K + kg*4 + r]   for j in [0,N), r in [0,4)
 * K=90 has a 2-element tail (90 = 22*4 + 2) -> 23 k-groups total, tail zero-padded.
 *
 * Per output row i: Q6_Vw_vrmpyacc_VwVbVb K-reduction (A splatted, B varies by
 * lane j) -> int32 accumulators; scalar epilogue (bias add, saturate, LUT).
 * N=8 is small so the scalar epilogue is cheap and keeps bit-exact bias/LUT.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, const int8_t *gelu_lut,
                      int8_t *out, int M, int N, int K)
{
    int num_full = K >> 2;                   /* 22 */
    int tail     = K & 3;                    /* 2  */
    int num_kgroups = num_full + (tail > 0 ? 1 : 0); /* 23 */

    /* Bt: num_kgroups x 128 bytes; word lane j (0..N-1, N<=32 so one block
     * per k-group suffices) holds {B[j][k0..k0+3]}. */
    static int8_t __attribute__((aligned(128))) Bt[23 * 128];

    for (int kg = 0; kg < num_full; kg++) {
        int8_t * restrict blk = Bt + kg * 128;
        int k0 = kg * 4;
        for (int j = 0; j < N; j++) {
            blk[j*4+0] = B[j*K + k0+0];
            blk[j*4+1] = B[j*K + k0+1];
            blk[j*4+2] = B[j*K + k0+2];
            blk[j*4+3] = B[j*K + k0+3];
        }
        for (int j = N; j < 32; j++) {
            blk[j*4+0] = blk[j*4+1] = blk[j*4+2] = blk[j*4+3] = 0;
        }
    }
    if (tail > 0) {
        int8_t *blk = Bt + num_full * 128;
        int k0 = num_full * 4;
        memset(blk, 0, 128);
        for (int j = 0; j < N; j++)
            for (int t = 0; t < tail; t++)
                blk[j*4+t] = B[j*K + k0+t];
    }

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;
        HVX_Vector acc = Q6_V_vzero();

        for (int kg = 0; kg < num_kgroups; kg++) {
            int k0 = kg * 4;
            int32_t a4;
            if (kg < num_full) {
                a4 = (int32_t)((uint8_t)Arow[k0+0])
                   | (int32_t)((uint8_t)Arow[k0+1] <<  8)
                   | (int32_t)((uint8_t)Arow[k0+2] << 16)
                   | (int32_t)((uint8_t)Arow[k0+3] << 24);
            } else {
                a4 = 0;
                for (int t = 0; t < tail; t++)
                    a4 |= (int32_t)((uint8_t)Arow[k0+t]) << (t*8);
            }
            HVX_Vector va = Q6_V_vsplat_R(a4);
            HVX_Vector vb = *(const HVX_Vector *)(Bt + kg * 128);
            acc = Q6_Vw_vrmpyacc_VwVbVb(acc, va, vb);
        }

        int32_t accbuf[32] __attribute__((aligned(128)));
        *(HVX_Vector *)accbuf = acc;

        int8_t *out_row = out + i * N;
        for (int j = 0; j < N; j++) {
            int64_t biased = (int64_t)accbuf[j] + (int64_t)bias[j];
            if (biased >  127) biased =  127;
            if (biased < -128) biased = -128;
            int8_t pre = (int8_t)biased;
            uint8_t idx = (uint8_t)(pre + 128);
            out_row[j] = gelu_lut[idx];
        }
    }
}
