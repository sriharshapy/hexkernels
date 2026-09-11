/* Near-miss: applies GELU correctly in layer 1 but then returns hid[] directly
   instead of computing layer 2 (GEMM2 + b2 + requant). Output shape is wrong
   conceptually; this compiles but writes garbage to out[] and fails. */
#include <stdint.h>

static int8_t sat8_sg(int32_t x) {
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
    (void)W2; (void)b2; (void)mult; (void)shift; (void)zp; /* BUG: second GEMM skipped */

    /* Layer 1 only -- correct GELU */
    for (int i = 0; i < M; i++) {
        for (int v = 0; v < V && v < D; v++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)W1[k*V+v];
            acc += b1[v];
            int8_t pre = sat8_sg(acc);
            uint8_t idx = (uint8_t)(pre + 128);
            /* BUG: writes hid into out without doing GEMM2 */
            out[i*D+v] = gelu_lut[idx];
        }
    }
}
