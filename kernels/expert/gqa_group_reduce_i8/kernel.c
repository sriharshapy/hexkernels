/* EXPERT (achievability bar) -- same as solutions/s2.c: pairwise-tree
 * reduction STRUCTURE (marginally faster than s1's sequential chain at
 * this size). Instead of a sequential left-to-right accumulate chain,
 * reduces the HPG sign-
 * extended head-plane vectors PAIRWISE in a tree (X0+X1, X2+X3, ...,
 * then sum the partial sums, halving the count each level) until one
 * result remains. Assumes HPG is a power of two (true for the pinned
 * H=8,G=2,HPG=4 configuration). Same lo/hi (even/odd) deinterleave
 * convention as s1 for Q6_Wh_vsxt_Vb, and the same scalar requant
 * epilogue writing back through the matching interleave. Arithmetically
 * identical to a sequential sum (16-bit add is associative here, no
 * overflow since HPG*127 is tiny), but a structurally distinct
 * dependency chain / instruction schedule.
 *
 * Head-plane stride is MN=136 bytes, which is NOT a multiple of 128, so a
 * head plane's base pointer (X + head*MN) is generally NOT 128B-aligned --
 * a raw `*(const HVX_Vector *)` load from it would silently read from the
 * wrong (rounded-down) address. Each 128-byte chunk is therefore staged
 * through a small 128B-aligned local scratch buffer before being loaded
 * as an HVX_Vector. */
#include "kernel_api.h"
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static int8_t requant_i8(int32_t acc, int32_t mult, int shift) {
    int64_t r    = (int64_t)acc * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q    = (r >= 0) ? ((r + half) >> shift) : -((-r + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

void candidate_kernel(const int8_t *X, int8_t *Y, int H, int G, int M, int N,
                      int32_t scale_mult, int scale_shift) {
    int MN  = M * N;
    int HPG = H / G;

    int16_t evenbuf[64] __attribute__((aligned(128)));
    int16_t oddbuf[64]  __attribute__((aligned(128)));
    HVX_Vector treeLo[64], treeHi[64]; /* generous upper bound on HPG */
    uint8_t chunk[64][128] __attribute__((aligned(128))); /* per-plane staging */

    for (int g = 0; g < G; g++) {
        const int8_t *planes[64];
        for (int t = 0; t < HPG; t++) planes[t] = X + (long)(g*HPG + t) * MN;
        int8_t *Yg = Y + (long)g * MN;

        int base = 0;
        for (; base + 128 <= MN; base += 128) {
            for (int t = 0; t < HPG; t++) {
                memcpy(chunk[t], planes[t] + base, 128);
                HVX_VectorPair pt = Q6_Wh_vsxt_Vb(*(const HVX_Vector *)chunk[t]);
                treeLo[t] = Q6_V_lo_W(pt);
                treeHi[t] = Q6_V_hi_W(pt);
            }
            int count = HPG;
            while (count > 1) {
                int half_n = count / 2;
                for (int i = 0; i < half_n; i++) {
                    treeLo[i] = Q6_Vh_vadd_VhVh(treeLo[2*i], treeLo[2*i + 1]);
                    treeHi[i] = Q6_Vh_vadd_VhVh(treeHi[2*i], treeHi[2*i + 1]);
                }
                count = half_n;
            }
            *(HVX_Vector *)evenbuf = treeLo[0];
            *(HVX_Vector *)oddbuf  = treeHi[0];

            for (int j = 0; j < 64; j++) {
                Yg[base + 2*j]     = requant_i8((int32_t)evenbuf[j], scale_mult, scale_shift);
                Yg[base + 2*j + 1] = requant_i8((int32_t)oddbuf[j],  scale_mult, scale_shift);
            }
        }
        /* scalar tail */
        for (int idx = base; idx < MN; idx++) {
            int32_t acc = 0;
            for (int t = 0; t < HPG; t++) acc += (int32_t)planes[t][idx];
            Yg[idx] = requant_i8(acc, scale_mult, scale_shift);
        }
    }
}
