/*
 * i8_depthwise_bias_relu_1d HVX kernel
 *
 * Pinned: C=8, L=256, ntaps=7, xstride=262.
 *
 * Strategy: 64-output FIR blocks, using Q6_Ww_vmpyacc_WwVhVh (even/odd split)
 * then Q6_W_vshuff_VVR(-4) to reorder to sequential int32 order.
 *
 * Pack pipeline: int32 → int16 via Q6_Vh_vpack_VwVw_sat (SEQUENTIAL, NOT vsat)
 * then int16 → int8 via Q6_Vb_vpack_VhVh_sat.
 *
 * Two 64-output blocks combined into one 128-byte aligned store using
 * valign to merge duplicate-tailed pA with pB.
 *
 * Key finding: Q6_Vh_vsat_VwVw INTERLEAVES (even/odd), while
 * Q6_Vh_vpack_VwVw_sat CONCATENATES (sequential). Use the latter.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

/* Load 128 contiguous bytes from unaligned address p */
static inline HVX_Vector valign_load(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    return Q6_V_valign_VVR(*(vp + 1), *vp, (int)((uintptr_t)p & 127));
}

/*
 * FIR for 64 outputs starting at position i in channel xch.
 * Returns a VectorPair with SEQUENTIAL int32 layout:
 *   result.lo = FIR(i..i+31), result.hi = FIR(i+32..i+63)
 */
static inline HVX_VectorPair fir_64(
        const int8_t *xch, const int8_t *tapch, int i, int ntaps)
{
    HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
    for (int j = 0; j < ntaps; j++) {
        HVX_Vector raw  = valign_load(xch + i + j);
        /* Sequential unpack: lo.h[k] = x[i+j+k] for k=0..63 */
        HVX_Vector lo16 = Q6_V_lo_W(Q6_Wh_vunpack_Vb(raw));
        HVX_Vector vtap = Q6_Vh_vsplat_R((int32_t)(int16_t)(int8_t)tapch[j]);
        /* Even/odd accumulation:
           acc.lo[k] += lo16.h[2k] * vtap = x[i+j+2k] * tap[j] → even positions
           acc.hi[k] += lo16.h[2k+1] * vtap = x[i+j+2k+1] * tap[j] → odd positions */
        acc = Q6_Ww_vmpyacc_WwVhVh(acc, lo16, vtap);
    }
    /* Interleave even/odd to sequential order:
       sorted.lo = [FIR(i), FIR(i+1), ..., FIR(i+31)]
       sorted.hi = [FIR(i+32), ..., FIR(i+63)] */
    return Q6_W_vshuff_VVR(Q6_V_hi_W(acc), Q6_V_lo_W(acc), -4);
}

/*
 * Pack 64 int32 values → 64 int8 values.
 * w0 = positions 0..31 (sequential int32), w1 = positions 32..63.
 * Returns 64 int8 in bytes 0..63, with DUPLICATE in bytes 64..127.
 *
 * Uses Q6_Vh_vpack_VwVw_sat (SEQUENTIAL pack, NOT vsat which is interleaved).
 */
static inline HVX_Vector pack64(HVX_Vector w0, HVX_Vector w1,
                                HVX_Vector vbias, HVX_Vector vzero, HVX_Vector v127)
{
    /* Apply bias + ReLU + clamp to [0,127] */
    w0 = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vadd_VwVw(w0, vbias), vzero), v127);
    w1 = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vadd_VwVw(w1, vbias), vzero), v127);

    /* int32 → int16 (sequential: h[0..31]=w0, h[32..63]=w1) */
    HVX_Vector pack_h = Q6_Vh_vpack_VwVw_sat(w1, w0);  /* Vu=w1, Vv=w0 → Vv in lower half */

    /* int16 → int8 (b[0..63]=sat(h[0..63]), b[64..127]=duplicate) */
    return Q6_Vb_vpack_VhVh_sat(pack_h, pack_h);
}

void candidate_kernel(const int8_t *x, const int8_t *taps, const int32_t *bias,
                      int8_t *out, int L, int C, int ntaps) {
    const int xstride = L + ntaps - 1;  /* 262 */

    HVX_Vector vzero = Q6_V_vzero();
    HVX_Vector v127  = Q6_V_vsplat_R(127);

    for (int ch = 0; ch < C; ch++) {
        const int8_t *xch   = x    + ch * xstride;
        const int8_t *tapch = taps + ch * ntaps;
        int8_t       *och   = out  + ch * L;
        HVX_Vector    vbias = Q6_V_vsplat_R(bias[ch]);

        /* L=256 outputs: 2 groups of 128. Each group = 2 sub-blocks of 64. */
        for (int grp = 0; grp < 2; grp++) {
            int i = grp * 128;

            /* FIR for positions i..i+63 */
            HVX_VectorPair sA = fir_64(xch, tapch, i,      ntaps);
            /* FIR for positions i+64..i+127 */
            HVX_VectorPair sB = fir_64(xch, tapch, i + 64, ntaps);

            /* Pack each 64-output block to int8 (with duplicate in upper half) */
            HVX_Vector pA = pack64(Q6_V_lo_W(sA), Q6_V_hi_W(sA), vbias, vzero, v127);
            HVX_Vector pB = pack64(Q6_V_lo_W(sB), Q6_V_hi_W(sB), vbias, vzero, v127);

            /*
             * Combine pA and pB into 128-byte aligned output.
             * pA.b[0..63] = positions i..i+63 (valid)
             * pA.b[64..127] = positions i..i+63 (duplicate, also valid)
             * pB.b[0..63] = positions i+64..i+127 (valid)
             *
             * valign_VVR(pB, pA, 64):
             *   result.b[k] = pA.b[k+64] for k<64 = positions i..i+63 ✓
             *   result.b[k] = pB.b[k-64] for k≥64 = positions i+64..i+127 ✓
             */
            HVX_Vector out128 = Q6_V_valign_VVR(pB, pA, 64);

            /* Aligned store: och+i is 128B-aligned for i=0,128 */
            *(HVX_Vector *)(och + i) = out128;
        }
    }
}
