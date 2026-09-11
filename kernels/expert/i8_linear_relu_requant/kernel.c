/* i8_linear_relu_requant sol_02: HVX vrmpy for K=128 dot product.
 * K=128 fits in one HVX vector (no tail). Scalar requant+relu. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline int32_t hsum_vw(HVX_Vector v) {
    int32_t *p = (int32_t *)&v;
    int32_t s = 0;
    for (int l = 0; l < 32; l++) s += p[l];
    return s;
}

void candidate_kernel(const int8_t *A, const int8_t *x,
                      const int32_t *bias, int8_t *out,
                      int M, int K,
                      int32_t mult, int shift, int8_t zp) {
    HVX_Vector vx = *(const HVX_Vector *)x;  /* K=128 */

    for (int i = 0; i < M; i++) {
        HVX_Vector vA = *(const HVX_Vector *)(A + i * K);
        int32_t acc = hsum_vw(Q6_Vw_vrmpy_VbVb(vA, vx));
        /* scalar tail for any K > 128 */
        for (int k = 128; k < K; k++)
            acc += (int32_t)A[i*K+k] * (int32_t)x[k];

        int64_t biased = (int64_t)acc + (int64_t)bias[i];
        if (biased < 0) biased = 0;  /* ReLU */

        int64_t v    = biased * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += (int64_t)zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}