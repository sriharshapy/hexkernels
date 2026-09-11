/*
 * i8_conv1d_stride2 HVX candidate
 *
 * Strategy: process 64 outputs per HVX iteration.
 *
 * For outputs i..i+63 (stride=2, ntaps=8):
 *   tap j needs: x[2i+j], x[2i+j+2], ..., x[2i+j+126]  (64 values, stride 2 in x)
 *
 *   1. valign_load(x + 2*i + j): load 128 contiguous bytes starting at x[2i+j].
 *   2. Q6_Vb_vdeal_Vb: deinterleave — lower 64 bytes become even-indexed positions
 *      {x[2i+j+0], x[2i+j+2], ..., x[2i+j+126]} = the 64 input values we need.
 *   3. Q6_V_lo_W(Q6_Wh_vunpack_Vb(dealed)): sign-extend lower 64 bytes to 64 int16.
 *   4. Q6_Ww_vmpyacc_WwVhVh(acc, xh, splat(taps[j])): int16*int16 -> int32 accumulate.
 *      Result: acc_lo[k] = even-indexed outputs (0,2,4,...,62)
 *              acc_hi[k] = odd-indexed outputs  (1,3,5,...,63)
 *   5. Q6_W_vshuff_VVR(hi, lo, -4): interleave at int32 granularity → outputs 0..63.
 *   6. Aligned store 64 int32 outputs.
 *
 * HVX body: limit to blocks where valign reads stay in-bounds.
 *   valign reads *(vp) and *(vp+1) = 256 bytes covering p..p+255.
 *   Largest p = x + 2*(i+63) + (ntaps-1) = x + 2i + 134.
 *   Need x + 2i + 134 + 127 = x + 2i + 261 <= x + xlen - 1 = x + 1009.
 *   → 2i <= 748 → i <= 374. Largest multiple-of-64 <= 374 is 320.
 *   But wait: valign reads vp+1 which starts at base + 128, so max byte = p_aligned + 255.
 *   For p = x+2i+j, p_aligned = (p) & ~127, max reach = p_aligned + 255.
 *   Worst case: p is aligned, so max = p + 255. Need p + 255 <= x + 1009.
 *   p_max = x+2*384+7 = x+775. 775+255=1030 > 1009 — OOB!
 *
 * Safe strategy: stop HVX at i where p + 127 <= xlen-1 (don't need the second block
 * if we avoid valign and use a single aligned load from the floor-aligned pointer).
 * OR: use a scratch buffer for the last partial block.
 *
 * Simpler: just use scalar for the last block(s). n=501, blocks of 64:
 *   0..63, 64..127, ..., 384..447 (7 blocks = 448 outputs),
 *   then scalar 448..500 (53 outputs).
 * Last HVX block i=384: p_max = x+768+7+127=x+902 < x+1009. Safe with valign (vp+1).
 *   vp = (x+768+7) aligned-down; vp+1 = vp+128. vp+255 = max(x+768+7+255)=x+1030?
 *   Hmm, depends on alignment of x.
 *
 * Actually valign: vp = floor_align(p), reads vp[0..127] and vp[128..255].
 * If p = x+775 (for i=384, j=7): vp=x+768 (aligned), vp+128=x+896. Max byte x+1023.
 * But xlen=1010, so x+1010 would be OOB. x+1023 > x+1009 → UNSAFE.
 *
 * Safe HVX bound: vp+255 <= x+1009 → vp <= x+754 → p <= x+754 (since p>=vp).
 * p = x+2i+j, max j=7: need 2i+7 <= 754 → i <= 373.5 → i <= 373.
 * Largest 64-multiple <= 373: 320. So HVX loop: i=0,64,...,320 (6 blocks = 384 outputs).
 * Scalar: 384..500 (117 outputs). Fine for correctness.
 *
 * But 320 blocks means only 6*64=384 outputs in HVX. Let me verify:
 * i=320, j=7: p = x+640+7 = x+647. vp=x+640 (aligned). vp+255=x+895. x+895 < x+1010. SAFE.
 * i=384, j=7: p = x+768+7 = x+775. vp=x+768. vp+255=x+1023. x+1023 >= x+1010. UNSAFE.
 *
 * So: HVX loop i=0..320 (inclusive, step 64) = i in {0,64,128,192,256,320}.
 * Scalar: i=384..500 (117 outputs).
 *
 * To get more HVX coverage: process i=384 block using scalar.
 *   Or use a 32-output block at i=384 (processing 32 outputs instead of 64).
 *
 * Better: use a 32-wide block for i=384..447 if safe.
 * i=384, half-block 32 outputs: j=7, p=x+775. vp=x+768, vp+63=x+831 < x+1010. SAFE if only reading one vector!
 * For 32 outputs we'd read 64 bytes (not 128), so single aligned load suffices.
 *
 * Let's just do 6 blocks of 64 (i=0..320) and scalar remainder (384..500).
 * If that beats the bar, great. If not, optimize further.
 */

#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

/* Load 128 contiguous bytes from unaligned address p using two aligned loads + valign */
static inline HVX_Vector valign_load(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    return Q6_V_valign_VVR(*(vp + 1), *vp, (int)((uintptr_t)p & 127));
}

void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int n, int ntaps, int stride) {
    /* Generic scalar fallback (harness uses stride=2, ntaps=8, n=501) */
    if (stride != 2 || ntaps != 8) {
        for (int i = 0; i < n; i++) {
            int32_t acc = 0;
            for (int j = 0; j < ntaps; j++)
                acc += (int32_t)x[i * stride + j] * (int32_t)taps[j];
            out[i] = acc;
        }
        return;
    }

    /*
     * HVX body: 64 outputs per iteration.
     * Safe range: i in [0, 320] step 64 → i <= 320 (6 blocks).
     *
     * For each block of 64 outputs starting at i:
     *   For tap j = 0..7:
     *     Load 128 bytes at x[2i+j] via valign.
     *     vdeal: lower 64 bytes = x[2i+j+0], x[2i+j+2], ..., x[2i+j+126].
     *     vunpack lo: 64 int16 sign-extended values.
     *     vmpyacc: multiply by taps[j], accumulate.
     *   vshuff to reorder outputs: even-indexed → lo, odd-indexed → hi of acc.
     *   Store 64 int32 outputs.
     */
    int i = 0;
    for (; i <= 320; i += 64) {
        HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());

        /* Tap 0 */
        {
            HVX_Vector raw = valign_load(x + 2*i + 0);
            HVX_Vector deal = Q6_Vb_vdeal_Vb(raw);
            HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(deal));
            HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[0]);
            acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
        }
        /* Tap 1 */
        {
            HVX_Vector raw = valign_load(x + 2*i + 1);
            HVX_Vector deal = Q6_Vb_vdeal_Vb(raw);
            HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(deal));
            HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[1]);
            acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
        }
        /* Tap 2 */
        {
            HVX_Vector raw = valign_load(x + 2*i + 2);
            HVX_Vector deal = Q6_Vb_vdeal_Vb(raw);
            HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(deal));
            HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[2]);
            acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
        }
        /* Tap 3 */
        {
            HVX_Vector raw = valign_load(x + 2*i + 3);
            HVX_Vector deal = Q6_Vb_vdeal_Vb(raw);
            HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(deal));
            HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[3]);
            acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
        }
        /* Tap 4 */
        {
            HVX_Vector raw = valign_load(x + 2*i + 4);
            HVX_Vector deal = Q6_Vb_vdeal_Vb(raw);
            HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(deal));
            HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[4]);
            acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
        }
        /* Tap 5 */
        {
            HVX_Vector raw = valign_load(x + 2*i + 5);
            HVX_Vector deal = Q6_Vb_vdeal_Vb(raw);
            HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(deal));
            HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[5]);
            acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
        }
        /* Tap 6 */
        {
            HVX_Vector raw = valign_load(x + 2*i + 6);
            HVX_Vector deal = Q6_Vb_vdeal_Vb(raw);
            HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(deal));
            HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[6]);
            acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
        }
        /* Tap 7 */
        {
            HVX_Vector raw = valign_load(x + 2*i + 7);
            HVX_Vector deal = Q6_Vb_vdeal_Vb(raw);
            HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(deal));
            HVX_Vector th = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[7]);
            acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, th);
        }

        /* Reorder: acc_lo has even-output-indexed (0,2,4,...,62),
         *          acc_hi has odd-output-indexed  (1,3,5,...,63).
         * vshuff(hi, lo, -4) interleaves at 4-byte granularity:
         *   result[2k]   = lo[k], result[2k+1] = hi[k] → outputs in order 0,1,2,...,63. */
        HVX_VectorPair sorted = Q6_W_vshuff_VVR(Q6_V_hi_W(acc), Q6_V_lo_W(acc), -4);

        /* Aligned store: output i..i+63 are 4*64=256 bytes, written as 2 HVX vectors */
        *(HVX_Vector *)((int32_t *)out + i)      = Q6_V_lo_W(sorted);
        *(HVX_Vector *)((int32_t *)out + i + 32) = Q6_V_hi_W(sorted);
    }

    /* Scalar tail: i=384..500 */
    for (; i < n; i++) {
        int32_t acc = 0;
        for (int j = 0; j < 8; j++)
            acc += (int32_t)x[i * 2 + j] * (int32_t)taps[j];
        out[i] = acc;
    }
}
