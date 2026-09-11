/*
 * u8_l2pool2x2 — HVX candidate
 *
 * W=130, H=98 → OW=65, OH=49
 * Semantics: out[oy*OW+ox] = clamp255((uint32_t)sqrtf(a*a+b*b+c*c+d*d))
 *   where a=in[2oy*W+2ox], b=in[2oy*W+2ox+1],
 *         c=in[(2oy+1)*W+2ox], d=in[(2oy+1)*W+2ox+1]
 *
 * Strategy per output row (64 output pixels via HVX, 1 tail pixel scalar):
 *
 *  1. Load 128 bytes from input row r0 (even row) and r1 (odd row) via __builtin_memcpy.
 *  2. Q6_W_vdeal_VVR(vzero, v, -1): separate even/odd column bytes.
 *     .lo[k] = v[2k]   for k=0..63  → a[] or c[]
 *     .hi[k] = v[2k+1] for k=0..63  → b[] or d[]
 *  3. Q6_Wuh_vunpack_Vub: widen u8→u16 for the 64-byte lo/hi halves.
 *     (Only .lo of the result matters; .hi holds widened zeros from the
 *      upper half of the vdeal output.)
 *  4. Q6_Ww_vmpy_VhVh(v16, v16): square u16→i32 for all 64 values.
 *     Result VectorPair: .lo = even-lane i32 squares, .hi = odd-lane i32 squares.
 *  5. Sum all 4 contributions: sum_lo += a_sq.lo + b_sq.lo + c_sq.lo + d_sq.lo
 *                               sum_hi += a_sq.hi + b_sq.hi + c_sq.hi + d_sq.hi
 *  6. Store sum_lo (32 i32) and sum_hi (32 i32) to temp arrays.
 *     sum_lo[k] = a[2k]²+b[2k]²+c[2k]²+d[2k]² for k=0..31
 *     sum_hi[k] = a[2k+1]²+b[2k+1]²+c[2k+1]²+d[2k+1]² for k=0..31
 *  7. Scalar sqrtf() and clamp, write output in correct order:
 *     out[2k] from sum_lo[k], out[2k+1] from sum_hi[k].
 *  8. Tail: scalar for output col 64 (input cols 128,129).
 */

#include <stdint.h>
#include <math.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector hvx_loadu(const uint8_t *p) {
    HVX_Vector v;
    __builtin_memcpy(&v, p, 128);
    return v;
}

void candidate_kernel(const uint8_t * restrict in, uint8_t * restrict out,
                      int w, int h)
{
    int ow = w >> 1;   /* 65 */
    int oh = h >> 1;   /* 49 */
    HVX_Vector vzero = Q6_V_vzero();

    for (int oy = 0; oy < oh; oy++) {
        const uint8_t *r0 = in + (2 * oy)     * w;
        const uint8_t *r1 = in + (2 * oy + 1) * w;
        uint8_t       *op = out + oy * ow;

        /* ---- HVX body: 64 output pixels ---- */

        /* Load 128 input bytes from each row */
        HVX_Vector v0 = hvx_loadu(r0);
        HVX_Vector v1 = hvx_loadu(r1);

        /* Deinterleave: separate even/odd column bytes
         * Q6_W_vdeal_VVR(Vu=vzero, Vv=vx, R=-1):
         *   result.lo[k] = vx[2k]   for k=0..63  (even input cols → a or c pixels)
         *   result.hi[k] = vx[2k+1] for k=0..63  (odd  input cols → b or d pixels)
         * Upper 64 bytes of lo/hi are from vzero → 0.
         */
        HVX_VectorPair d0 = Q6_W_vdeal_VVR(vzero, v0, -1);
        HVX_VectorPair d1 = Q6_W_vdeal_VVR(vzero, v1, -1);

        /* Extract separated halves (each: 64 real bytes in 0..63, zeros in 64..127) */
        HVX_Vector a_u8 = Q6_V_lo_W(d0);  /* a[0..63] = r0[0,2,...,126] */
        HVX_Vector b_u8 = Q6_V_hi_W(d0);  /* b[0..63] = r0[1,3,...,127] */
        HVX_Vector c_u8 = Q6_V_lo_W(d1);  /* c[0..63] = r1[0,2,...,126] */
        HVX_Vector d_u8 = Q6_V_hi_W(d1);  /* d[0..63] = r1[1,3,...,127] */

        /* Widen u8→u16: Q6_Wuh_vunpack_Vub
         * For a 128-byte input where bytes 0..63 hold our data and 64..127 are zero:
         *   result.lo: bytes 0..63 widened → 64 u16 (our 64 values)
         *   result.hi: bytes 64..127 widened → 64 zeros (discarded)
         */
        HVX_Vector a_u16 = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(a_u8));
        HVX_Vector b_u16 = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(b_u8));
        HVX_Vector c_u16 = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(c_u8));
        HVX_Vector d_u16 = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(d_u8));

        /* Square u16→i32: Q6_Ww_vmpy_VhVh(Vu, Vv)
         * Interprets each 128-byte vector as 64 i16 halfwords.
         * Returns VectorPair:
         *   .lo[k] = Vu[2k] * Vv[2k]   for k=0..31  (even halfwords → i32)
         *   .hi[k] = Vu[2k+1] * Vv[2k+1] for k=0..31 (odd halfwords → i32)
         * Since our u16 values are 0..255, they fit in i16 as positive values.
         * Squaring: max 255²=65025 fits in i32 positive.
         */
        HVX_VectorPair sq_a = Q6_Ww_vmpy_VhVh(a_u16, a_u16);  /* a² */
        HVX_VectorPair sq_b = Q6_Ww_vmpy_VhVh(b_u16, b_u16);  /* b² */
        HVX_VectorPair sq_c = Q6_Ww_vmpy_VhVh(c_u16, c_u16);  /* c² */
        HVX_VectorPair sq_d = Q6_Ww_vmpy_VhVh(d_u16, d_u16);  /* d² */

        /* Accumulate sums: sum_lo[k] = a[2k]²+b[2k]²+c[2k]²+d[2k]²  (max=260100)
         *                  sum_hi[k] = a[2k+1]²+b[2k+1]²+c[2k+1]²+d[2k+1]²
         * These fit in i32 (max 260100 << 2^31).
         */
        HVX_Vector sum_lo = Q6_Vw_vadd_VwVw(
                                Q6_Vw_vadd_VwVw(Q6_V_lo_W(sq_a), Q6_V_lo_W(sq_b)),
                                Q6_Vw_vadd_VwVw(Q6_V_lo_W(sq_c), Q6_V_lo_W(sq_d)));
        HVX_Vector sum_hi = Q6_Vw_vadd_VwVw(
                                Q6_Vw_vadd_VwVw(Q6_V_hi_W(sq_a), Q6_V_hi_W(sq_b)),
                                Q6_Vw_vadd_VwVw(Q6_V_hi_W(sq_c), Q6_V_hi_W(sq_d)));

        /* Store 32+32 i32 sums to temp buffer for scalar sqrt */
        int32_t sums_e[32] __attribute__((aligned(128)));  /* even pixels */
        int32_t sums_o[32] __attribute__((aligned(128)));  /* odd pixels  */
        *(HVX_Vector *)sums_e = sum_lo;
        *(HVX_Vector *)sums_o = sum_hi;

        /* Scalar sqrtf + clamp, write output in correct order:
         * out[2k]   = sqrt(sums_e[k])  (even output columns)
         * out[2k+1] = sqrt(sums_o[k])  (odd  output columns)
         */
        for (int k = 0; k < 32; k++) {
            uint32_t ve = (uint32_t)sqrtf((float)sums_e[k]);
            uint32_t vo = (uint32_t)sqrtf((float)sums_o[k]);
            op[2*k]   = (uint8_t)(ve > 255 ? 255 : ve);
            op[2*k+1] = (uint8_t)(vo > 255 ? 255 : vo);
        }

        /* ---- scalar tail: output col 64 (input bytes [128,129]) ---- */
        {
            int a = r0[128], b = r0[129];
            int c = r1[128], d = r1[129];
            uint32_t sv = (uint32_t)sqrtf((float)(a*a + b*b + c*c + d*d));
            op[64] = (uint8_t)(sv > 255 ? 255 : sv);
        }
    }
}
