/*
 * HVX A.V tile (column-major V), row-pair-packed vrmpy broadcast --
 * genuinely different algorithm from s1 (no per-element hreduce; instead
 * accumulates 16 output-columns at a time across 2 A-rows via vrmpy, same
 * structural idiom as the sibling qkt task's tile-broadcast expert).
 *
 * N=16 reduction -> exactly 4 k-groups of 4 (no reduction tail).
 * D=90 output columns -> processed in blocks of <=16 columns (6 blocks:
 * 5 full + 1 tail of 10), since one row-pair-packed vrmpy produces 32
 * accumulator lanes = 16 output-columns x 2 rows.
 *
 * Per block, build Kt[kg][128] from V (column-major: V[d*N+j] contiguous
 * over j for fixed d, so building the per-kg 4-byte-interleaved block is a
 * gather over 16 rows, not a big-stride transpose):
 *   Kt[kg*128 + j*4 + r]      = V[(bstart+j)*N + kg*4+r]   (lanes 0..15)
 *   Kt[kg*128 + 64 + j*4 + r] = V[(bstart+j)*N + kg*4+r]   (lanes 16..31, dup)
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(const int8_t *A, const int8_t *V, int32_t *O,
                      int M, int N, int D)
{
    int num_kgroups = N >> 2;   /* N=16 -> 4, exact (no reduction tail) */

    /* Precompute per-row 4-byte-packed A words for all k-groups, once. */
    static uint32_t awords[16][4];
    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * N;
        for (int kg = 0; kg < num_kgroups; kg++) {
            int k0 = kg * 4;
            awords[i][kg] = (uint32_t)((uint8_t)Arow[k0+0])
                          | ((uint32_t)((uint8_t)Arow[k0+1]) << 8)
                          | ((uint32_t)((uint8_t)Arow[k0+2]) << 16)
                          | ((uint32_t)((uint8_t)Arow[k0+3]) << 24);
        }
    }

    static int8_t __attribute__((aligned(128))) Kt[4 * 128];
    int32_t accbuf[32] __attribute__((aligned(128)));

    for (int bstart = 0; bstart < D; bstart += 16) {
        int bcount = D - bstart;
        if (bcount > 16) bcount = 16;

        for (int kg = 0; kg < num_kgroups; kg++) {
            int8_t *blk = Kt + kg * 128;
            int k0 = kg * 4;
            for (int j = 0; j < bcount; j++) {
                int8_t b0 = V[(bstart+j)*N + k0+0];
                int8_t b1 = V[(bstart+j)*N + k0+1];
                int8_t b2 = V[(bstart+j)*N + k0+2];
                int8_t b3 = V[(bstart+j)*N + k0+3];
                blk[j*4+0] = b0; blk[j*4+1] = b1; blk[j*4+2] = b2; blk[j*4+3] = b3;
                blk[64+j*4+0] = b0; blk[64+j*4+1] = b1; blk[64+j*4+2] = b2; blk[64+j*4+3] = b3;
            }
            for (int j = bcount; j < 16; j++) {
                blk[j*4+0]=blk[j*4+1]=blk[j*4+2]=blk[j*4+3]=0;
                blk[64+j*4+0]=blk[64+j*4+1]=blk[64+j*4+2]=blk[64+j*4+3]=0;
            }
        }

        for (int ip = 0; ip < M; ip += 2) {
            int i0 = ip, i1 = ip + 1;
            HVX_Vector acc = Q6_V_vzero();

            for (int kg = 0; kg < num_kgroups; kg++) {
                uint32_t w0 = awords[i0][kg];
                uint32_t w1 = awords[i1][kg];
                HVX_Vector vA = Q6_V_vsplat_R(w0);
                HVX_Vector vB = Q6_V_vsplat_R(w1);
                HVX_Vector va = Q6_V_valign_VVR(vB, vA, 64);
                HVX_Vector vb = *(const HVX_Vector *)(Kt + kg * 128);
                acc = Q6_Vw_vrmpyacc_VwVbVb(acc, va, vb);
            }

            *(HVX_Vector *)accbuf = acc;

            int32_t *Orow0 = O + i0 * D + bstart;
            int32_t *Orow1 = O + i1 * D + bstart;
            for (int j = 0; j < bcount; j++) {
                Orow0[j] = accbuf[j];
                Orow1[j] = accbuf[16 + j];
            }
        }
    }
}
