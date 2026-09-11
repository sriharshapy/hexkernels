/* Near-miss: skips the GELU activation (layer 1 outputs sat8(acc+b1) directly,
   bypassing the LUT lookup). Layer 2 and requant are correct.
   Fails wherever GELU(x) != x (i.e. almost everywhere for large |acc|). */
#include <stdint.h>

static int8_t sat8_ng(int32_t x) {
    if (x >  127) return  127;
    if (x < -128) return -128;
    return (int8_t)x;
}

void candidate_kernel(const int8_t  *A,
                      const int8_t  *W1, const int32_t *b1,
                      const int8_t  *gelu_lut,
                      const int8_t  *W2, const int32_t *b2,
                      int8_t        *out,
                      int M, int K, int V, int D,
                      int32_t mult, int shift, int8_t zp) {
    (void)gelu_lut; /* BUG: LUT ignored -- GELU step missing */
    int8_t hid[8*32];

    /* Layer 1 without GELU -- just sat8(acc+b1) */
    for (int i = 0; i < M; i++) {
        for (int v = 0; v < V; v++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)W1[k*V+v];
            acc += b1[v];
            hid[i*V+v] = sat8_ng(acc);  /* BUG: no GELU */
        }
    }

    /* Layer 2: correct */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) {
            int32_t acc = 0;
            for (int v = 0; v < V; v++)
                acc += (int32_t)hid[i*V+v] * (int32_t)W2[v*D+j];
            int64_t biased = (int64_t)acc + (int64_t)b2[j];
            int64_t vv   = biased * (int64_t)mult;
            int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
            int64_t r    = (vv >= 0) ? ((vv + half) >> shift) : -(((-vv) + half) >> shift);
            r += zp;
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            out[i*D+j] = (int8_t)r;
        }
    }
}
