/*
 * HVX FFN: fc1(K=16)->GELU->fc2(D=16) + bias + residual + LayerNorm (integer).
 * M=8 tokens, K=16, V=32, D=16.
 *
 * Strategy:
 *   Layer 1 GEMM (M x K) x (K x V):
 *     - K=16 -> 4 k-groups of 4; V=32 -> one 128-byte HVX vector per row.
 *     - Pretranspose W1: 4 groups x 128B.
 *     - vrmpy for K-reduction, scalar bias+sat8+GELU-LUT epilogue.
 *
 *   Layer 2 GEMM (M x V) x (V x D):
 *     - V=32 -> 8 v-groups of 4; D=16 -> 16 active int32 lanes (pad to 32).
 *     - Pretranspose W2: 8 groups x 128B.
 *     - vrmpy for V-reduction, scalar bias+residual+sat8 epilogue.
 *
 *   LayerNorm: fully scalar (D=16 per token, 8 tokens -- very cheap).
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

static int8_t sat8_c(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

void candidate_kernel(const int8_t   *x,
                      const int8_t   *W1, const int32_t *b1,
                      const int8_t   *gelu_lut,
                      const int8_t   *W2, const int32_t *b2,
                      const int8_t   *x_res,
                      const int8_t   *gamma, const int8_t *beta,
                      const uint8_t  *inv_lut,
                      int8_t         *out,
                      int M, int K, int V, int D)
{
    /* === Pretranspose W1: K/4=4 k-groups x 128B (V=32 word lanes) === */
    /* Each 128B block: word lane v holds bytes {W1[k0*V+v], W1[k1*V+v], W1[k2*V+v], W1[k3*V+v]} */
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

    /* === Pretranspose W2: V/4=8 v-groups x 128B (D=16 active word lanes, rest 0) === */
    /* Each 128B block: word lane j (j<D) holds {W2[v0*D+j], W2[v1*D+j], W2[v2*D+j], W2[v3*D+j]} */
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

    /* Intermediate buffers */
    static int8_t  __attribute__((aligned(128))) hid[8 * 32]; /* M*V */
    static int8_t  __attribute__((aligned(128))) res[8 * 16]; /* M*D */

    int32_t accbuf[32] __attribute__((aligned(128)));

    /* ===================== Layer 1: GEMM + b1 + sat8 + GELU LUT ===================== */
    /* Load b1 as HVX vector (V=32 int32 words = 128B) */
    HVX_Vector vb1 = *(const HVX_Vector *)b1;

    for (int i = 0; i < M; i++) {
        const int8_t *Arow = x + i * K;
        HVX_Vector acc = Q6_V_vzero();

        /* 4 k-groups, each 4 input elements x 32 output elements */
#define KG1(kg, base)                                               \
        {                                                           \
            int32_t a4 = (int32_t)((uint8_t)Arow[(base)+0])        \
                       | ((int32_t)((uint8_t)Arow[(base)+1]) <<  8) \
                       | ((int32_t)((uint8_t)Arow[(base)+2]) << 16) \
                       | ((int32_t)((uint8_t)Arow[(base)+3]) << 24); \
            HVX_Vector va = Q6_V_vsplat_R(a4);                      \
            HVX_Vector vw = *(const HVX_Vector *)(Bt1 + (kg) * 128); \
            acc = Q6_Vw_vrmpyacc_VwVbVb(acc, va, vw);               \
        }

        KG1(0,  0)
        KG1(1,  4)
        KG1(2,  8)
        KG1(3, 12)
#undef KG1

        /* Add bias b1 */
        HVX_Vector biased = Q6_Vw_vadd_VwVw(acc, vb1);
        *(HVX_Vector *)accbuf = biased;

        /* Scalar sat8 + GELU LUT */
        int8_t *hrow = hid + i * V;
        for (int v = 0; v < V; v++) {
            int8_t pre = sat8_c(accbuf[v]);
            uint8_t idx = (uint8_t)(pre + 128);
            hrow[v] = gelu_lut[idx];
        }
    }

    /* ===================== Layer 2: GEMM + b2 + residual + sat8 ===================== */
    for (int i = 0; i < M; i++) {
        const int8_t *hrow = hid + i * V;
        HVX_Vector acc2 = Q6_V_vzero();

        /* 8 v-groups, each 4 hidden elements x D=16 output elements */
#define KG2(kg, base)                                                \
        {                                                            \
            int32_t h4 = (int32_t)((uint8_t)hrow[(base)+0])         \
                       | ((int32_t)((uint8_t)hrow[(base)+1]) <<  8)  \
                       | ((int32_t)((uint8_t)hrow[(base)+2]) << 16)  \
                       | ((int32_t)((uint8_t)hrow[(base)+3]) << 24); \
            HVX_Vector vh = Q6_V_vsplat_R(h4);                       \
            HVX_Vector vw2 = *(const HVX_Vector *)(Bt2 + (kg) * 128); \
            acc2 = Q6_Vw_vrmpyacc_VwVbVb(acc2, vh, vw2);             \
        }

        KG2(0,  0)
        KG2(1,  4)
        KG2(2,  8)
        KG2(3, 12)
        KG2(4, 16)
        KG2(5, 20)
        KG2(6, 24)
        KG2(7, 28)
#undef KG2

        *(HVX_Vector *)accbuf = acc2;

        /* Scalar bias + residual add + sat8 (only D=16 lanes valid) */
        int8_t *rrow = res + i * D;
        const int8_t *xres_row = x_res + i * D;
        for (int j = 0; j < D; j++) {
            int64_t ffn_out = (int64_t)accbuf[j] + (int64_t)b2[j];
            int64_t added   = ffn_out + (int32_t)xres_row[j];
            rrow[j] = sat8_c((int32_t)added);
        }
    }

    /* ===================== LayerNorm per token (scalar, D=16 small) ===================== */
    for (int i = 0; i < M; i++) {
        const int8_t *rrow = res + i * D;
        int8_t *orow = out + i * D;

        /* Mean */
        int32_t sum = 0;
        for (int j = 0; j < D; j++) sum += (int32_t)rrow[j];
        int32_t mu = sum / D;

        /* Variance */
        int32_t var_sum = 0;
        for (int j = 0; j < D; j++) {
            int32_t d = (int32_t)rrow[j] - mu;
            var_sum += d * d;
        }
        int32_t var = var_sum / D;
        int32_t v_idx = var;
        if (v_idx < 0)   v_idx = 0;
        if (v_idx > 255) v_idx = 255;
        uint8_t inv = inv_lut[(uint8_t)v_idx];

        /* Normalize */
        for (int j = 0; j < D; j++) {
            int32_t d      = (int32_t)rrow[j] - mu;
            int32_t scaled = (d * (int32_t)gamma[j] + 64) >> 7;
            int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
            int32_t r      = normed + (int32_t)beta[j];
            orow[j] = sat8_c(r);
        }
    }
}
