/* sol_02: i8_conv1d_depthwise_requant â€” HVX per-channel vectorised FIR + scalar requant.
 * FIR accumulation is vectorised (int16*int16->int32 vmpyacc), requant stays scalar
 * because int64 multiply is hard to vectorize correctly. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector valign_v(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    return Q6_V_valign_VVR(*(vp+1), *vp, (int)((uintptr_t)p & 127));
}

static inline int8_t requant(int32_t acc, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int C, int L, int K,
                      int32_t mult, int shift, int8_t zp) {
    for (int c = 0; c < C; c++) {
        const int8_t *xc   = x    + c * (L + K - 1);
        const int8_t *tapc = taps + c * K;
        int8_t       *oc   = out  + c * L;

        /* HVX: compute K-tap FIR for 64 outputs at a time, then scalar requant */
        int n_vec = (L / 64) * 64;
        int i = 0;
        for (; i < n_vec; i += 64) {
            HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
            for (int k = 0; k < K; k++) {
                HVX_Vector xb = valign_v(xc + i + k);
                HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(xb));
                HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)tapc[k]);
                acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
            }
            /* Unpack the int32 results from the VectorPair */
            HVX_VectorPair ord = Q6_W_vshuff_VVR(Q6_V_hi_W(acc), Q6_V_lo_W(acc), -4);
            __attribute__((aligned(128))) int32_t buf[64];
            *(HVX_Vector *)buf       = Q6_V_lo_W(ord);
            *(HVX_Vector *)(buf+32)  = Q6_V_hi_W(ord);
            for (int ii = 0; ii < 64; ii++)
                oc[i + ii] = requant(buf[ii], mult, shift, zp);
        }
        for (; i < L; i++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)xc[i + k] * (int32_t)tapc[k];
            oc[i] = requant(acc, mult, shift, zp);
        }
    }
}