/* sol_03: Direct 3x3 conv + requant. Im2col + HVX dot product.
 * Gathers the 9*C_in window, then dots with each weight row using
 * Q6_Vw_vrmpyacc_VwVbVb for HVX acceleration. */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

static inline int8_t requant_s64(int32_t acc, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* HVX int8 dot product, n must be multiple of 128 (caller pads) */
static int32_t dot128(const int8_t *a, const int8_t *b, int n) {
    int32_t total = 0;
    for (int i = 0; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        HVX_Vector vr = Q6_Vw_vrmpyacc_VwVbVb(Q6_V_vzero(), va, vb);
        const int32_t *p = (const int32_t *)&vr;
        for (int k = 0; k < 32; k++) total += p[k];
    }
    for (int i = (n / 128) * 128; i < n; i++)
        total += (int32_t)a[i] * (int32_t)b[i];
    return total;
}

void candidate_kernel(const int8_t *in, const int8_t *wt, int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp) {
    int win_size = 9 * C_in;
    /* Pad to multiple of 128 */
    int win_pad = (win_size + 127) / 128 * 128;
    int8_t win[win_pad + 128];
    int8_t *wp_a = (int8_t *)(((uintptr_t)win + 127) & ~(uintptr_t)127);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            /* Gather window */
            int idx = 0;
            for (int ky = 0; ky < 3; ky++) {
                int sy = y + ky - 1;
                for (int kx = 0; kx < 3; kx++) {
                    int sx = x + kx - 1;
                    for (int ci = 0; ci < C_in; ci++) {
                        wp_a[idx++] = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                                      ? in[sy*W*C_in + sx*C_in + ci] : (int8_t)0;
                    }
                }
            }
            memset(wp_a + idx, 0, win_pad - idx);

            for (int co = 0; co < C_out; co++) {
                /* Pad weight row */
                int8_t wb[win_pad + 128];
                int8_t *wp_b = (int8_t *)(((uintptr_t)wb + 127) & ~(uintptr_t)127);
                memcpy(wp_b, wt + co * win_size, win_size);
                memset(wp_b + win_size, 0, win_pad - win_size);

                int32_t acc = dot128(wp_a, wp_b, win_pad);
                out[y*W*C_out + x*C_out + co] = requant_s64(acc, mult, shift, zp);
            }
        }
    }
}