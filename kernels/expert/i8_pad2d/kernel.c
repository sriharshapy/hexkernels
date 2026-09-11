/* sol_03: HVX zero-fill output, then scalar copy interior */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static void hvx_memzero(int8_t *dst, int len) {
    HVX_Vector zero = Q6_V_vzero();
    int i = 0;
    for (; i + 128 <= len; i += 128)
        *(HVX_Vector *)(dst + i) = zero;
    for (; i < len; i++)
        dst[i] = 0;
}

void candidate_kernel(const int8_t *in, int8_t *out,
                      int H, int W, int P) {
    int OH = H + 2 * P;
    int OW = W + 2 * P;
    hvx_memzero(out, OH * OW);
    /* Copy interior pixel rows */
    for (int r = 0; r < H; r++) {
        const int8_t *src = in + r * W;
        int8_t *dst = out + (r + P) * OW + P;
        for (int c = 0; c < W; c++)
            dst[c] = src[c];
    }
}