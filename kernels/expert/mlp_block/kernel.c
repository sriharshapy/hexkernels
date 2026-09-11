/*
 * HVX fused two-layer MLP block: fc1(K=16,V=32) + GELU-LUT + fc2(V=32,D=16)
 *
 * Strategy:
 *   Layer 1 (A[M*K] x W1[K*V] + b1[V]):
 *     - K=16 -> 4 k-groups of 4. V=32 output cols fit in ONE 128B HVX vector (32 int32 lanes).
 *     - Pretranspose W1 into Bt1[4 groups][128B]: word lane v = {W1[k0][v]..W1[k3][v]}
 *     - For each token i: vrmpy acc (32 lanes), vector bias add, scalar sat8+LUT -> hid[i*V+v]
 *
 *   Layer 2 (hid[M*V] x W2[V*D] + b2[D]):
 *     - V=32 -> 8 k-groups of 4. D=16 output cols fit in 16 int32 lanes (half a vector).
 *     - Pretranspose W2 into Bt2[8 groups][64B]: word lane d = {W2[v0][d]..W2[v3][d]}
 *     - For each token i: vrmpy acc (16 lanes), vector bias add, scalar requant -> out[i*D+j]
 *
 *   The GEMM accumulations are fully vectorized with Q6_Vw_vrmpyacc_VwVbVb.
 *   Scalar epilogues (sat8+LUT, requant) are cheap for these small output dims.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

void candidate_kernel(const int8_t  *A,
                      const int8_t  *W1, const int32_t *b1,
                      const int8_t  *gelu_lut,
                      const int8_t  *W2, const int32_t *b2,
                      int8_t        *out,
                      int M, int K, int V, int D,
                      int32_t mult, int shift, int8_t zp)
{
    /*
     * Bt1: K=16 / 4 = 4 k-groups, each 128B (V=32 word lanes).
     * Word lane v in group kg: {W1[k0][v], W1[k1][v], W1[k2][v], W1[k3][v]}
     */
    int8_t Bt1[4 * 128] __attribute__((aligned(128)));

    /*
     * Bt2: V=32 / 4 = 8 k-groups, each 64B (D=16 word lanes).
     * We store each group in 64 bytes (16 words = 64 bytes), padded to 128B alignment.
     * Word lane d in group kg: {W2[v0][d], W2[v1][d], W2[v2][d], W2[v3][d]}
     */
    int8_t Bt2[8 * 128] __attribute__((aligned(128)));

    int num_kg1 = K >> 2;  /* K/4 = 4 */
    int num_kg2 = V >> 2;  /* V/4 = 8 */

    /* Build transposed W1 (K*V -> k-groups * V) */
    for (int kg = 0; kg < num_kg1; kg++) {
        int8_t * restrict blk = Bt1 + kg * 128;
        const int k0 = kg * 4;
        for (int v = 0; v < V; v++) {
            blk[v*4+0] = W1[(k0+0)*V + v];
            blk[v*4+1] = W1[(k0+1)*V + v];
            blk[v*4+2] = W1[(k0+2)*V + v];
            blk[v*4+3] = W1[(k0+3)*V + v];
        }
    }

    /* Build transposed W2 (V*D -> k-groups * D), pad unused lanes to 0 */
    for (int kg = 0; kg < num_kg2; kg++) {
        int8_t * restrict blk = Bt2 + kg * 128;
        const int v0 = kg * 4;
        memset(blk, 0, 128);
        for (int d = 0; d < D; d++) {
            blk[d*4+0] = W2[(v0+0)*D + d];
            blk[d*4+1] = W2[(v0+1)*D + d];
            blk[d*4+2] = W2[(v0+2)*D + d];
            blk[d*4+3] = W2[(v0+3)*D + d];
        }
    }

    /* Requant constants */
    int64_t half_val = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int32_t izp = (int32_t)zp;

    /* Hidden buffer (int8, M*V = 8*32 = 256 bytes) */
    int8_t hid[8 * 32] __attribute__((aligned(128)));

    /* Accumulation scratch buffers */
    int32_t acc1buf[32] __attribute__((aligned(128)));  /* V=32 lanes for layer 1 */
    int32_t acc2buf[32] __attribute__((aligned(128)));  /* 32 lanes, only D=16 used */

    /* --- Layer 1 --- */
    for (int i = 0; i < M; i++) {
        const int8_t *Arow = A + i * K;

        HVX_Vector vacc = Q6_V_vzero(); /* 32 int32 word lanes for V=32 */

        for (int kg = 0; kg < num_kg1; kg++) {
            const int k0 = kg * 4;
            /* Pack Arow[k0..k3] as int32 word, splat across all lanes */
            int32_t a4 = (int32_t)((uint8_t)Arow[k0+0])
                       | (int32_t)((uint8_t)Arow[k0+1] <<  8)
                       | (int32_t)((uint8_t)Arow[k0+2] << 16)
                       | (int32_t)((uint8_t)Arow[k0+3] << 24);
            HVX_Vector va = Q6_V_vsplat_R(a4);
            HVX_Vector vb = *(const HVX_Vector *)(Bt1 + kg * 128);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, va, vb);
        }

        /* Vector bias add: bias[0..31] */
        HVX_Vector vbias = *(const HVX_Vector *)b1;  /* b1[0..31] */
        HVX_Vector vbiased = Q6_Vw_vadd_VwVw(vacc, vbias);

        /* Store to scalar buffer */
        *(HVX_Vector *)acc1buf = vbiased;

        /* Scalar epilogue: sat8 + GELU LUT */
        int8_t *hid_row = hid + i * V;
        for (int v = 0; v < V; v++) {
            int32_t b = acc1buf[v];
            if (b >  127) b =  127;
            if (b < -128) b = -128;
            int8_t pre = (int8_t)b;
            uint8_t idx = (uint8_t)(pre + 128);
            hid_row[v] = gelu_lut[idx];
        }
    }

    /* --- Layer 2 --- */
    for (int i = 0; i < M; i++) {
        const int8_t *hrow = hid + i * V;

        HVX_Vector vacc = Q6_V_vzero(); /* 32 int32 word lanes; only first D=16 are valid */

        for (int kg = 0; kg < num_kg2; kg++) {
            const int v0 = kg * 4;
            /* Pack hrow[v0..v3] as int32 word, splat */
            int32_t h4 = (int32_t)((uint8_t)hrow[v0+0])
                       | (int32_t)((uint8_t)hrow[v0+1] <<  8)
                       | (int32_t)((uint8_t)hrow[v0+2] << 16)
                       | (int32_t)((uint8_t)hrow[v0+3] << 24);
            HVX_Vector vh = Q6_V_vsplat_R(h4);
            HVX_Vector vb = *(const HVX_Vector *)(Bt2 + kg * 128);
            vacc = Q6_Vw_vrmpyacc_VwVbVb(vacc, vh, vb);
        }

        /* Vector bias add: b2[0..15] in word lanes 0..15 */
        /* Load 16 int32s (64 bytes) = half a vector.
         * Use aligned load of full vector -- b2 is HVX_ALIGN so first 128B is valid.
         * Only lanes 0..15 matter; upper lanes don't matter since we only read acc2buf[0..15]. */
        HVX_Vector vbias2 = *(const HVX_Vector *)b2;  /* b2[0..31] -- only [0..15] used */
        HVX_Vector vbiased = Q6_Vw_vadd_VwVw(vacc, vbias2);

        /* Store to scalar buffer */
        *(HVX_Vector *)acc2buf = vbiased;

        /* Scalar requant epilogue */
        int8_t *out_row = out + i * D;
        for (int j = 0; j < D; j++) {
            int64_t bv = (int64_t)acc2buf[j];
            int64_t vv = bv * (int64_t)mult;
            int64_t r;
            if (vv >= 0) r = (vv + half_val) >> shift;
            else         r = -((-vv + half_val) >> shift);
            r += izp;
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            out_row[j] = (int8_t)r;
        }
    }
}
