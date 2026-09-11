/* Pure scalar int8 linear projection + bias + saturating requant-to-int8
 * (bit-exact) -- the DENOMINATOR baseline. W is stored [Dout,Din] row-major --
 * row o IS output-channel o's weight vector, so X[i,:] . W[o,:] is a direct
 * row-row dot product (no transpose needed). The same HMX-0x40-requant ->
 * bias -> saturate epilogue as the expert is applied per output element (no
 * ReLU -- this is a plain projection). */
#include "kernel_api.h"
#include "harness_common.h"

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

static inline int8_t saturate_i8(int v) {
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int8_t)v;
}

void candidate_kernel(const uint8_t *X, const int8_t *W, const int32_t *bias,
                       int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int o = 0; o < n; o++) {
            int acc = 0;
            for (int d = 0; d < n; d++) acc += (int)X[i*n+d] * (int)W[o*n+d];
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[o];
            out[i*n + o] = saturate_i8(biased);
        }
    }
}
