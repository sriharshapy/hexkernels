/*
 * HVX two-layer FFN: fc1(K=16)->ReLU+sat8->fc2(D=16) with bias+requant.
 * M=8 tokens, K=16, V=32, D=16.
 *
 * Strategy:
 *   Layer 1: vrmpy for GEMM, scalar sat8/relu for correctness.
 *   Layer 2: vrmpy for GEMM, scalar requant.
 *
 * Pretranspose W1 into 4 k-groups (128B each), W2 into 8 v-groups (128B each).
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

static int8_t sat8_s(int32_t x) {
    if (x >  127) return  127;
    if (x < -128) return -128;
    return (int8_t)x;
}

void candidate_kernel(const int8_t  *A,
                      const int8_t  *W1, const int32_t *b1,
                      const int8_t  *W2, const int32_t *b2,
                      int8_t        *out,
                      int M, int K, int V, int D,
                      int32_t mult, int shift, int8_t zp)
{
    /* === Pretranspose W1: K/4=4 k-groups x 128B === */
    static int8_t __attribute__((aligned(128))) Bt1[4 * 128];
    {
        for (int kg = 0; kg < 4; kg++) {
            int8_t * restrict blk = Bt1 + kg * 128;
            const int k0 = kg * 4;
            for (int v = 0; v < V; v++) {
                blk[v*4+0] = W1[(k0+0)*V+v];
                blk[v*4+1] = W1[(k0+1)*V+v];
                blk[v*4+2] = W1[(k0+2)*V+v];
                blk[v*4+3] = W1[(k0+3)*V+v];
            }
        }
    }

    /* === Pretranspose W2: V/4=8 v-groups x 128B (D=16 word lanes valid) === */
    static int8_t __attribute__((aligned(128))) Bt2[8 * 128];
    {
        for (int kg = 0; kg < 8; kg++) {
            int8_t * restrict blk = Bt2 + kg * 128;
            const int v0 = kg * 4;
            memset(blk, 0, 128);
            for (int j = 0; j < D; j++) {
                blk[j*4+0] = W2[(v0+0)*D+j];
                blk[j*4+1] = W2[(v0+1)*D+j];
                blk[j*4+2] = W2[(v0+2)*D+j];
                blk[j*4+3] = W2[(v0+3)*D+j];
            }
        }
    }

    /* Hidden buffer aligned for HVX */
    static int8_t __attribute__((aligned(128))) hid[8 * 32];

    /* Accumulation buffer (32 words) aligned for vector store/load */
    int32_t accbuf[32] __attribute__((aligned(128)));

    HVX_Vector vzero = Q6_V_vzero();
    HVX_Vector vb1   = *(const HVX_Vector *)b1; /* b1[0..31] = 128B */

    /* ===================== Layer 1 ===================== */
    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;
        HVX_Vector acc = Q6_V_vzero();

        /* 4 k-groups unrolled */
        int32_t a4;
        HVX_Vector va, vw;

#define KGROUP1(kg, base)                                           \
        a4 = (int32_t)((uint8_t)Arow[base+0])                      \
           | ((int32_t)((uint8_t)Arow[base+1]) <<  8)              \
           | ((int32_t)((uint8_t)Arow[base+2]) << 16)              \
           | ((int32_t)((uint8_t)Arow[base+3]) << 24);             \
        va = Q6_V_vsplat_R(a4);                                     \
        vw = *(const HVX_Vector *)(Bt1 + (kg) * 128);              \
        acc = Q6_Vw_vrmpyacc_VwVbVb(acc, va, vw)

        KGROUP1(0,  0);
        KGROUP1(1,  4);
        KGROUP1(2,  8);
        KGROUP1(3, 12);
#undef KGROUP1

        /* Add b1 */
        HVX_Vector biased = Q6_Vw_vadd_VwVw(acc, vb1);

        /* Store int32 biased to accbuf for scalar relu+sat8 */
        *(HVX_Vector *)accbuf = biased;

        int8_t *hrow = hid + i * V;
        for (int v = 0; v < V; v++) {
            int32_t bv = accbuf[v];
            if (bv < 0) bv = 0;   /* ReLU */
            hrow[v] = sat8_s(bv);
        }
    }

    /* ===================== Layer 2 ===================== */
    int64_t half_val = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int32_t izp = (int32_t)zp;

    for (int i = 0; i < M; i++) {
        const int8_t *hrow = hid + i * V;
        HVX_Vector acc2 = Q6_V_vzero();

        /* 8 v-groups unrolled */
        int32_t h4;
        HVX_Vector vh2, vw2;

#define KGROUP2(kg, base)                                           \
        h4 = (int32_t)((uint8_t)hrow[base+0])                      \
           | ((int32_t)((uint8_t)hrow[base+1]) <<  8)              \
           | ((int32_t)((uint8_t)hrow[base+2]) << 16)              \
           | ((int32_t)((uint8_t)hrow[base+3]) << 24);             \
        vh2 = Q6_V_vsplat_R(h4);                                    \
        vw2 = *(const HVX_Vector *)(Bt2 + (kg) * 128);             \
        acc2 = Q6_Vw_vrmpyacc_VwVbVb(acc2, vh2, vw2)

        KGROUP2(0,  0);
        KGROUP2(1,  4);
        KGROUP2(2,  8);
        KGROUP2(3, 12);
        KGROUP2(4, 16);
        KGROUP2(5, 20);
        KGROUP2(6, 24);
        KGROUP2(7, 28);
#undef KGROUP2

        /* Store D=16 int32 words for scalar requant */
        *(HVX_Vector *)accbuf = acc2;

        int8_t *orow = out + i * D;
        for (int j = 0; j < D; j++) {
            int64_t biased2 = (int64_t)accbuf[j] + (int64_t)b2[j];
            int64_t vv = biased2 * (int64_t)mult;
            int64_t r;
            if (vv >= 0) r = (vv + half_val) >> shift;
            else         r = -((-vv + half_val) >> shift);
            r += izp;
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            orow[j] = (int8_t)r;
        }
    }
}
