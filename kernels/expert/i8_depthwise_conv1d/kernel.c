/* sol_02: i8_depthwise_conv1d â€” HVX per-channel FIR.
 * Aligns HVX body to 128-byte output boundaries per channel.
 * Uses Q6_V_vstu_variable for unaligned scalar tail stores (avoids aligned write crash).
 * Actually: use masked store Q6_V_vstu_variable or just require aligned start per channel.
 * For L=250, ch*L*4 may not be 128-byte aligned for ch>0. Use vmemu (unaligned vmem).
 * vmemu = V_vmemu_QRV in some SDKs; alternatively, just store scalar when unaligned. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector valign_v(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    return Q6_V_valign_VVR(*(vp+1), *vp, (int)((uintptr_t)p & 127));
}

static inline void store_hvx_unaligned(int32_t *dst, HVX_Vector v) {
    __attribute__((aligned(128))) int32_t tmp[32];
    *(HVX_Vector *)tmp = v;
    for (int k = 0; k < 32; k++) dst[k] = tmp[k];
}

void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int L, int C, int ntaps) {
    int xstride = L + ntaps - 1;
    for (int ch = 0; ch < C; ch++) {
        const int8_t *xch   = x    + ch * xstride;
        const int8_t *tapch = taps + ch * ntaps;
        int32_t      *och   = out  + ch * L;

        int n_vec = (L / 64) * 64;
        int i = 0;
        for (; i < n_vec; i += 64) {
            HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
            for (int j = 0; j < ntaps; j++) {
                HVX_Vector xb = valign_v(xch + i + j);
                HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(xb));
                HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)tapch[j]);
                acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
            }
            HVX_VectorPair ord = Q6_W_vshuff_VVR(Q6_V_hi_W(acc), Q6_V_lo_W(acc), -4);
            store_hvx_unaligned(och + i,      Q6_V_lo_W(ord));
            store_hvx_unaligned(och + i + 32, Q6_V_hi_W(ord));
        }
        for (; i < L; i++) {
            int32_t acc = 0;
            for (int j = 0; j < ntaps; j++)
                acc += (int32_t)xch[i + j] * (int32_t)tapch[j];
            och[i] = acc;
        }
    }
}