/* sol_02: i8_fir_bias_requant â€” HVX vectorised FIR accumulation + scalar requant.
 * ntaps=16; process 64 outputs per HVX block. Inner tap loop uses vmpyacc. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector valign_v(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    return Q6_V_valign_VVR(*(vp+1), *vp, (int)((uintptr_t)p & 127));
}

static inline int8_t do_requant(int64_t biased, int32_t mult, int shift, int8_t zp) {
    int64_t v    = biased * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) return  127;
    if (r < -128) return -128;
    return (int8_t)r;
}

void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t bias,
                      int8_t *out,
                      int n, int ntaps,
                      int32_t mult, int shift, int8_t zp) {
    int n_vec = (n / 64) * 64;
    int i = 0;
    for (; i < n_vec; i += 64) {
        HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        for (int k = 0; k < ntaps; k++) {
            HVX_Vector xb = valign_v(x + i + k);
            HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(xb));
            HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[k]);
            acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
        }
        HVX_VectorPair ord = Q6_W_vshuff_VVR(Q6_V_hi_W(acc), Q6_V_lo_W(acc), -4);
        __attribute__((aligned(128))) int32_t buf[64];
        *(HVX_Vector *)buf      = Q6_V_lo_W(ord);
        *(HVX_Vector *)(buf+32) = Q6_V_hi_W(ord);
        for (int ii = 0; ii < 64; ii++)
            out[i+ii] = do_requant((int64_t)buf[ii] + (int64_t)bias, mult, shift, zp);
    }
    for (; i < n; i++) {
        int32_t acc = 0;
        for (int k = 0; k < ntaps; k++)
            acc += (int32_t)x[i+k] * (int32_t)taps[k];
        out[i] = do_requant((int64_t)acc + (int64_t)bias, mult, shift, zp);
    }
}