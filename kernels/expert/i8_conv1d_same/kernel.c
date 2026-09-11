/* sol_02: i8_conv1d_same â€” HVX body aligned to 256-byte output boundary.
 * The body must start at an index i such that (int32_t*)out + i is 128-byte aligned
 * (i must be multiple of 32, since each int32 is 4 bytes, 32*4=128).
 * We align to the next multiple of 32 >= pad_left. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector valign_load(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    return Q6_V_valign_VVR(*(vp+1), *vp, (int)((uintptr_t)p & 127));
}

void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int n, int ntaps) {
    int pad_left = (ntaps - 1) / 2;

    /* Scalar pass for everything: left border, any misalignment, right border */
    /* First pass all scalar (body too) for correctness, then HVX where safe. */
    /* Actually: output array is aligned. To write aligned HVX vectors of int32,
     * we need i % 32 == 0 (since HVX_Vector = 32 int32). Align up from pad_left. */
    int hvx_start = pad_left;
    if (hvx_start % 32 != 0)
        hvx_start = hvx_start + (32 - hvx_start % 32);

    /* Upper bound: last access x[i - pad_left + ntaps-1 + 63] < n
     * So i < n - (ntaps-1) - 63 + pad_left */
    int hvx_end_safe = n - (ntaps - 1) - 63 + pad_left;
    /* Also end must be multiple of 64 above hvx_start */
    int hvx_end = hvx_start + ((hvx_end_safe - hvx_start) / 64) * 64;
    if (hvx_end < hvx_start) hvx_end = hvx_start;

    int i = 0;

    /* Scalar up to hvx_start */
    for (; i < hvx_start; i++) {
        int32_t acc = 0;
        for (int j = 0; j < ntaps; j++) {
            int xi = i - pad_left + j;
            if (xi >= 0 && xi < n)
                acc += (int32_t)x[xi] * (int32_t)taps[j];
        }
        out[i] = acc;
    }

    /* HVX body: aligned writes, no OOB reads */
    for (; i < hvx_end; i += 64) {
        HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        for (int j = 0; j < ntaps; j++) {
            HVX_Vector xb = valign_load(x + (i - pad_left) + j);
            HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(xb));
            HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[j]);
            acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
        }
        HVX_VectorPair ord = Q6_W_vshuff_VVR(Q6_V_hi_W(acc), Q6_V_lo_W(acc), -4);
        *(HVX_Vector *)((int32_t *)out + i)      = Q6_V_lo_W(ord);
        *(HVX_Vector *)((int32_t *)out + i + 32) = Q6_V_hi_W(ord);
    }

    /* Scalar tail */
    for (; i < n; i++) {
        int32_t acc = 0;
        for (int j = 0; j < ntaps; j++) {
            int xi = i - pad_left + j;
            if (xi >= 0 && xi < n)
                acc += (int32_t)x[xi] * (int32_t)taps[j];
        }
        out[i] = acc;
    }
}