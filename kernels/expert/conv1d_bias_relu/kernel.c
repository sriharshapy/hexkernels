/*
 * conv1d_bias_relu HVX candidate
 *
 * Shape (pinned): L=512, K=5, C=16 (channels-last / interleaved)
 * x layout:    x[sample * 16 + channel],   total 516*16 = 8256 bytes
 * taps layout: taps[k * 16 + channel],     total   5*16 =   80 bytes
 * out layout:  out[sample * 16 + channel], total 512*16 = 8192 bytes
 *
 * acc[i][c] = sum_{k=0}^{4} x[(i+k)*16+c] * taps[k*16+c]   (int32)
 * out[i][c] = sat8(relu(acc[i][c] + bias[c]))
 *
 * HVX strategy — process 8 output positions per iteration:
 *
 *   One 128B HVX vector holds 8 positions x 16 channels.
 *   For each tap k:
 *     1. Load 128B at x[(i+k)*16] (8 positions x C=16 channels).
 *     2. Build 128B broadcast tap vector (16-byte tap[k][*] repeated 8 times).
 *     3. Widen int8 → int16: Q6_Wh_vunpack_Vb gives sequential int16 (lo=bytes 0..63, hi=64..127).
 *     4. Q6_Ww_vmpyacc_WwVhVh accumulates int16*int16→int32.
 *        Result is deinterleaved (even-indexed in lo, odd-indexed in hi of pair).
 *   After K=5 taps:
 *     5. Q6_W_vshuff_VVR(hi, lo, -4): re-interleave at int32 granularity → sequential order.
 *
 *   Maintain 4 VectorPair accumulators for the 4 halves of the 8-position block.
 *   After accumulation: add bias, relu, pack int32→int8, store.
 *
 * Pack: Q6_Vh_vpack_VwVw_sat (SEQUENTIAL, not Q6_Vh_vsat_VwVw which interleaves).
 *       Q6_Vb_vpack_VhVh_sat (SEQUENTIAL).
 */

#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

/* Unaligned 128-byte load: reads 128 consecutive bytes starting at p. */
static inline HVX_Vector vload128(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    return Q6_V_valign_VVR(*(vp + 1), *vp, (int)((uintptr_t)p & 127));
}

void candidate_kernel(const int8_t *x, const int8_t *taps, const int32_t *bias,
                      int8_t *out,
                      int L, int K, int C)
{
    /* Scalar fallback for non-pinned shapes */
    if (C != 16 || K != 5) {
        for (int i = 0; i < L; i++) {
            for (int c = 0; c < C; c++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)x[(i+k)*C+c] * (int32_t)taps[k*C+c];
                int32_t v = acc + bias[c];
                if (v < 0)   v = 0;
                if (v > 127) v = 127;
                out[i*C+c] = (int8_t)v;
            }
        }
        return;
    }

    /*
     * Pre-broadcast K=5 tap vectors.
     * tap[k][c] is at taps[k*16 + c] (16 bytes per tap position).
     * Broadcast to 128B: repeat 8 times so tv[k][p*16+c] = taps[k*16+c].
     */
    HVX_Vector tv[5];
    {
        int8_t buf[128] __attribute__((aligned(128)));
        int32_t *b4 = (int32_t *)buf;
        for (int k = 0; k < 5; k++) {
            const int32_t *t4 = (const int32_t *)(taps + k * 16);
            for (int rep = 0; rep < 8; rep++) {
                b4[rep*4 + 0] = t4[0];
                b4[rep*4 + 1] = t4[1];
                b4[rep*4 + 2] = t4[2];
                b4[rep*4 + 3] = t4[3];
            }
            tv[k] = *(HVX_Vector *)buf;
        }
    }

    /*
     * Pre-widened tap vectors (int16), for use with Q6_Ww_vmpyacc_WwVhVh.
     * Q6_Wh_vunpack_Vb gives sequential sign-extension:
     *   lo = int16[0..63] from bytes 0..63
     *   hi = int16[0..63] from bytes 64..127
     * tw_lo[k] = int16 taps for positions 0-3, tw_hi[k] = for positions 4-7.
     * (Both halves are identical since tv[k] repeats the 16-byte tap.)
     */
    HVX_Vector tw_lo[5], tw_hi[5];
    for (int k = 0; k < 5; k++) {
        HVX_VectorPair tw = Q6_Wh_vunpack_Vb(tv[k]);
        tw_lo[k] = Q6_V_lo_W(tw);
        tw_hi[k] = Q6_V_hi_W(tw);
    }

    /*
     * Bias vectors: bias[0..15] repeated twice (for 2 positions per accumulator vector).
     * We will use these to build per-pair bias vectors after collecting sequential int32.
     * Actually build the bias as a 128B vector of 32 int32 values: [bias[0..15], bias[0..15]].
     */
    HVX_Vector bv;
    {
        int32_t bbuf[32] __attribute__((aligned(128)));
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 16; c++)
                bbuf[r*16 + c] = bias[c];
        bv = *(HVX_Vector *)bbuf;
    }

    HVX_Vector vzero = Q6_V_vzero();

    /* Process L=512 positions in blocks of 8. L=512 divisible by 8 → no tail. */
    for (int i = 0; i < L; i += 8) {
        /*
         * Accumulators in deinterleaved (even/odd) form.
         * Q6_Ww_vmpyacc_WwVhVh produces:
         *   acc_pair.lo[j] += vh_a[2j] * vh_b[2j]    (even-indexed products)
         *   acc_pair.hi[j] += vh_a[2j+1] * vh_b[2j+1] (odd-indexed products)
         *
         * We need 4 pairs for the 4 int16-halves of the 128-byte x block:
         *   pair01_lo: from xw_lo for bytes 0..63 (positions 0-3, all channels, even bytes)
         *   pair01_hi: from xw_lo for bytes 0..63 (odd bytes)
         *   pair23_lo: from xw_hi for bytes 64..127 (positions 4-7, all channels, even bytes)
         *   pair23_hi: from xw_hi for bytes 64..127 (odd bytes)
         *
         * Actually Q6_Ww_vmpyacc_WwVhVh takes one VectorPair accumulator for one Vh pair.
         * We operate on xw_lo (64 int16) and tw_lo (64 int16) → pair acc01 (64 int32, deinterleaved).
         * And on xw_hi (64 int16) and tw_hi (64 int16) → pair acc23 (64 int32, deinterleaved).
         */
        HVX_VectorPair acc01 = Q6_W_vcombine_VV(vzero, vzero);  /* for bytes 0..63 */
        HVX_VectorPair acc23 = Q6_W_vcombine_VV(vzero, vzero);  /* for bytes 64..127 */

        for (int k = 0; k < 5; k++) {
            /* Load 128B at x[(i+k)*16] — positions i..i+7, all 16 channels */
            HVX_Vector xv = vload128(x + (i + k) * 16);

            /* Sign-extend int8 → int16 (sequential) */
            HVX_VectorPair xw = Q6_Wh_vunpack_Vb(xv);
            /* xw_lo = int16 for bytes 0..63: positions 0-3, channels 0..15 */
            /* xw_hi = int16 for bytes 64..127: positions 4-7, channels 0..15 */

            /*
             * Q6_Ww_vmpyacc_WwVhVh(acc, Vh_a, Vh_b):
             *   acc.lo[j] += Vh_a[2j] * Vh_b[2j]     (even-indexed pairs)
             *   acc.hi[j] += Vh_a[2j+1] * Vh_b[2j+1] (odd-indexed pairs)
             * for j=0..31.
             * Result: deinterleaved int32 accumulation (even in .lo, odd in .hi).
             */
            acc01 = Q6_Ww_vmpyacc_WwVhVh(acc01, Q6_V_lo_W(xw), tw_lo[k]);
            acc23 = Q6_Ww_vmpyacc_WwVhVh(acc23, Q6_V_hi_W(xw), tw_hi[k]);
        }

        /*
         * Re-interleave deinterleaved int32 pairs to sequential order.
         * Q6_W_vshuff_VVR(Vu=hi, Vv=lo, Rt=-4) interleaves at int32 (4-byte) granularity:
         *   result[2j]   = lo[j]  → product at even index 2j
         *   result[2j+1] = hi[j]  → product at odd index 2j+1
         * → sequential int32 order.
         */
        HVX_VectorPair seq01 = Q6_W_vshuff_VVR(Q6_V_hi_W(acc01), Q6_V_lo_W(acc01), -4);
        HVX_VectorPair seq23 = Q6_W_vshuff_VVR(Q6_V_hi_W(acc23), Q6_V_lo_W(acc23), -4);
        /*
         * seq01.lo = int32 for bytes 0..31 = positions 0-1, channels 0..15 (32 int32)
         * seq01.hi = int32 for bytes 32..63 = positions 2-3, channels 0..15 (32 int32)
         * seq23.lo = int32 for bytes 64..95 = positions 4-5, channels 0..15 (32 int32)
         * seq23.hi = int32 for bytes 96..127 = positions 6-7, channels 0..15 (32 int32)
         */

        /* Add bias (bv = [bias[0..15], bias[0..15]] for 2 positions) */
        HVX_Vector s01_lo = Q6_Vw_vadd_VwVw(Q6_V_lo_W(seq01), bv);
        HVX_Vector s01_hi = Q6_Vw_vadd_VwVw(Q6_V_hi_W(seq01), bv);
        HVX_Vector s23_lo = Q6_Vw_vadd_VwVw(Q6_V_lo_W(seq23), bv);
        HVX_Vector s23_hi = Q6_Vw_vadd_VwVw(Q6_V_hi_W(seq23), bv);

        /* ReLU: max(v, 0) */
        s01_lo = Q6_Vw_vmax_VwVw(s01_lo, vzero);
        s01_hi = Q6_Vw_vmax_VwVw(s01_hi, vzero);
        s23_lo = Q6_Vw_vmax_VwVw(s23_lo, vzero);
        s23_hi = Q6_Vw_vmax_VwVw(s23_hi, vzero);

        /*
         * Pack int32 → int8 (saturating, sequential).
         * Q6_Vh_vpack_VwVw_sat(Vu, Vv): Vv → lower 32 int16, Vu → upper 32 int16.
         * Q6_Vb_vpack_VhVh_sat(Vu, Vv): Vv → lower 64 int8, Vu → upper 64 int8.
         * After ReLU, values ≥ 0, so saturation → [0, 127].
         */
        HVX_Vector v01h = Q6_Vh_vpack_VwVw_sat(s01_hi, s01_lo);  /* pos 0-3, 64 int16 */
        HVX_Vector v23h = Q6_Vh_vpack_VwVw_sat(s23_hi, s23_lo);  /* pos 4-7, 64 int16 */
        HVX_Vector vout = Q6_Vb_vpack_VhVh_sat(v23h, v01h);       /* 128 int8, pos 0-7 */

        /* Aligned store: out + i*16 is 128B-aligned for i = 0, 8, 16, ... */
        *(HVX_Vector *)(out + i * 16) = vout;
    }

    /* Scalar tail (empty for L=512, but correct for any L) */
    for (int i = (L & ~7); i < L; i++) {
        for (int c = 0; c < 16; c++) {
            int32_t acc = 0;
            for (int k = 0; k < 5; k++)
                acc += (int32_t)x[(i+k)*16+c] * (int32_t)taps[k*16+c];
            int32_t v = acc + bias[c];
            if (v < 0)   v = 0;
            if (v > 127) v = 127;
            out[i*16+c] = (int8_t)v;
        }
    }
}
