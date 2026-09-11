/* Near-miss: skips the residual add (x_res ignored). The LayerNorm still runs
   but on the raw FFN output instead of FFN+residual. Fails wherever the residual
   changes the LayerNorm statistics. */
#include <stdint.h>

static int8_t sat8_sr(int32_t v) {
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
    (void)x_res; /* BUG: residual skipped */
    int8_t hid[8*32];
    int8_t res[8*16];

    for (int i = 0; i < M; i++) {
        for (int v = 0; v < V; v++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)x[i*K+k] * (int32_t)W1[k*V+v];
            acc += b1[v];
            int8_t pre = sat8_sr(acc);
            uint8_t idx = (uint8_t)(pre + 128);
            hid[i*V+v] = gelu_lut[idx];
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) {
            int32_t acc = 0;
            for (int v = 0; v < V; v++)
                acc += (int32_t)hid[i*V+v] * (int32_t)W2[v*D+j];
            int64_t ffn_out = (int64_t)acc + (int64_t)b2[j];
            /* BUG: no residual add */
            res[i*D+j] = sat8_sr((int32_t)ffn_out);
        }
    }

    for (int i = 0; i < M; i++) {
        int32_t sum = 0;
        for (int j = 0; j < D; j++) sum += (int32_t)res[i*D+j];
        int32_t mu = sum / D;
        int32_t var_sum = 0;
        for (int j = 0; j < D; j++) {
            int32_t d = (int32_t)res[i*D+j] - mu;
            var_sum += d * d;
        }
        int32_t var = var_sum / D;
        if (var < 0) var = 0; if (var > 255) var = 255;
        uint8_t inv = inv_lut[var];
        for (int j = 0; j < D; j++) {
            int32_t d      = (int32_t)res[i*D+j] - mu;
            int32_t scaled = (d * (int32_t)gamma[j] + 64) >> 7;
            int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
            out[i*D+j] = sat8_sr(normed + (int32_t)beta[j]);
        }
    }
}
