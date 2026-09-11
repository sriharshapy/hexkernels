/* Near-miss: hardcodes the GELU LUT as identity (ignores runtime gelu_lut).
   Also hardcodes the LayerNorm inv_lut as constant 128.
   Fails on virtually every output element. */
#include <stdint.h>

static int8_t sat8_hl(int32_t v) {
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
                      int M, int K, int V, int D) {
    (void)gelu_lut;  /* BUG: GELU LUT ignored, uses identity */
    (void)inv_lut;   /* BUG: inv LUT ignored, hardcoded constant */
    int8_t hid[8*32];
    int8_t res[8*16];

    for (int i = 0; i < M; i++) {
        for (int v = 0; v < V; v++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)x[i*K+k] * (int32_t)W1[k*V+v];
            acc += b1[v];
            /* BUG: identity activation instead of gelu_lut lookup */
            hid[i*V+v] = sat8_hl(acc);
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) {
            int32_t acc = 0;
            for (int v = 0; v < V; v++)
                acc += (int32_t)hid[i*V+v] * (int32_t)W2[v*D+j];
            int64_t ffn_out = (int64_t)acc + (int64_t)b2[j];
            int64_t added   = ffn_out + (int32_t)x_res[i*D+j];
            res[i*D+j] = sat8_hl((int32_t)added);
        }
    }

    for (int i = 0; i < M; i++) {
        int32_t sum = 0;
        for (int j = 0; j < D; j++) sum += (int32_t)res[i*D+j];
        int32_t mu = sum / D;
        /* BUG: hardcoded inv=128 instead of reading inv_lut[var] */
        uint8_t inv = 128;
        for (int j = 0; j < D; j++) {
            int32_t d      = (int32_t)res[i*D+j] - mu;
            int32_t scaled = (d * (int32_t)gamma[j] + 64) >> 7;
            int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
            out[i*D+j] = sat8_hl(normed + (int32_t)beta[j]);
        }
    }
}
