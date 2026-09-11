/*
 * i8_gather_requant HVX kernel
 *
 * Gather : compress int32 indices → bytes, then Q6_Vb_vlut32/vlut32or
 *          (8 groups of 32 covering the full 256-entry table).
 * Requant: sign-expand i8→i16→i32, multiply by mult (product fits int32 since
 *          |raw|<=128, |mult|<=13), round-half-away-from-zero via abs+add+shift+
 *          restore-sign, add zp, saturate pack i32→i16→i8.
 *
 * Unpack semantics (verified from working kernels in this repo):
 *   Q6_Wh_vunpack_Vb(vb): 128 i8  → W (256B pair): lo=i16[0..63], hi=i16[64..127]
 *   Q6_Ww_vunpack_Vh(vh): 64  i16 → W (256B pair): lo=i32[0..31], hi=i32[32..63]
 *
 * So 128 i8 → 128 i32 in 3 unpack calls giving 4 vectors of 32 int32 each.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector loadv(const void *p) { return *(const HVX_Vector *)p; }
static inline void storev(void *p, HVX_Vector v) { *(HVX_Vector *)p = v; }

/* Requant 32 int32 lanes: round-half-away-from-zero, shift, add zp.
 * Result fits int32 so no overflow before the final saturating pack. */
static inline HVX_Vector rq32(HVX_Vector vprod,
                               HVX_Vector vhalf, HVX_Vector vzp, int sh) {
    HVX_Vector vzero = Q6_V_vzero();
    HVX_Vector vabs  = Q6_Vw_vabs_Vw(vprod);
    HVX_Vector vadd;
    if (sh > 0) {
        vadd = Q6_Vw_vadd_VwVw(vabs, vhalf);
        vadd = Q6_Vw_vasr_VwR(vadd, sh);
    } else {
        vadd = vabs;
    }
    /* Restore sign: where vprod < 0, negate */
    HVX_VectorPred qneg = Q6_Q_vcmp_gt_VwVw(vzero, vprod);
    HVX_Vector vneg     = Q6_Vw_vsub_VwVw(vzero, vadd);
    HVX_Vector vrnd     = Q6_V_vmux_QVV(qneg, vneg, vadd);
    return Q6_Vw_vadd_VwVw(vrnd, vzp);
}

void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {

    /* Load 256-byte table into two shuffled HVX vectors for vlut32.
     * Q6_Vb_vshuff_Vb reorders the table bytes into the layout expected by vlut32:
     * without shuffle, vlut32 reads the wrong offsets. */
    HVX_Vector tbl0 = Q6_Vb_vshuff_Vb(loadv(table));         /* entries   0..127, shuffled */
    HVX_Vector tbl1 = Q6_Vb_vshuff_Vb(loadv(table + 128));   /* entries 128..255, shuffled */

    /* Pack mult into both halfwords for Q6_Vw_vmpyi_VwRh
     * (Vd.w[i] = Vu.w[i] * Rt.h[i%2]; valid for |mult| <= 32767) */
    int32_t m_packed = (int32_t)((uint32_t)(uint16_t)(int16_t)mult |
                                  ((uint32_t)(uint16_t)(int16_t)mult << 16));
    int32_t half_val = (shift > 0) ? (1 << (shift - 1)) : 0;
    HVX_Vector vhalf = Q6_V_vsplat_R((uint32_t)half_val);
    HVX_Vector vzp   = Q6_V_vsplat_R((uint32_t)(int32_t)(int8_t)zp);
    int sh = shift;

    int i;
    for (i = 0; i + 128 <= n; i += 128) {
        const int32_t *ip = idx + i;

        /* Load 128 int32 indices → 4 HVX vectors (32 int32 each) */
        HVX_Vector vi0 = loadv(ip);
        HVX_Vector vi1 = loadv(ip + 32);
        HVX_Vector vi2 = loadv(ip + 64);
        HVX_Vector vi3 = loadv(ip + 96);

        /* Pack int32 → byte (low byte = index in [0,255]) */
        HVX_Vector vh01  = Q6_Vh_vpacke_VwVw(vi1, vi0);
        HVX_Vector vh23  = Q6_Vh_vpacke_VwVw(vi3, vi2);
        HVX_Vector vbidx = Q6_Vb_vpacke_VhVh(vh23, vh01);

        /* Gather 256-entry i8 table: vlut32 groups 0..7 */
        HVX_Vector vg;
        vg = Q6_Vb_vlut32_VbVbI(    vbidx, tbl0, 0);
        vg = Q6_Vb_vlut32or_VbVbVbI(vg, vbidx, tbl0, 1);
        vg = Q6_Vb_vlut32or_VbVbVbI(vg, vbidx, tbl0, 2);
        vg = Q6_Vb_vlut32or_VbVbVbI(vg, vbidx, tbl0, 3);
        vg = Q6_Vb_vlut32or_VbVbVbI(vg, vbidx, tbl1, 4);
        vg = Q6_Vb_vlut32or_VbVbVbI(vg, vbidx, tbl1, 5);
        vg = Q6_Vb_vlut32or_VbVbVbI(vg, vbidx, tbl1, 6);
        vg = Q6_Vb_vlut32or_VbVbVbI(vg, vbidx, tbl1, 7);

        /* Sign-expand: 128 i8 → 128 i16 in pair, then 64 i16 → 64 i32 in pair */
        HVX_VectorPair wp16  = Q6_Wh_vunpack_Vb(vg);
        HVX_VectorPair wp32a = Q6_Ww_vunpack_Vh(Q6_V_lo_W(wp16)); /* i32[0..63]  lo+hi */
        HVX_VectorPair wp32b = Q6_Ww_vunpack_Vh(Q6_V_hi_W(wp16)); /* i32[64..127] lo+hi */

        /* 4 vectors of 32 int32 lanes */
        HVX_Vector raw0 = Q6_V_lo_W(wp32a); /* i32[0..31]   */
        HVX_Vector raw1 = Q6_V_hi_W(wp32a); /* i32[32..63]  */
        HVX_Vector raw2 = Q6_V_lo_W(wp32b); /* i32[64..95]  */
        HVX_Vector raw3 = Q6_V_hi_W(wp32b); /* i32[96..127] */

        /* Multiply by mult (low 32 bits; all products fit int32 here) */
        HVX_Vector p0 = Q6_Vw_vmpyi_VwRh(raw0, m_packed);
        HVX_Vector p1 = Q6_Vw_vmpyi_VwRh(raw1, m_packed);
        HVX_Vector p2 = Q6_Vw_vmpyi_VwRh(raw2, m_packed);
        HVX_Vector p3 = Q6_Vw_vmpyi_VwRh(raw3, m_packed);

        /* Requant: round-half-away, shift, add zp */
        HVX_Vector r0 = rq32(p0, vhalf, vzp, sh);
        HVX_Vector r1 = rq32(p1, vhalf, vzp, sh);
        HVX_Vector r2 = rq32(p2, vhalf, vzp, sh);
        HVX_Vector r3 = rq32(p3, vhalf, vzp, sh);

        /* Saturating pack i32→i16→i8 */
        HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(r1, r0); /* 64 i16 [0..63]   */
        HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(r3, r2); /* 64 i16 [64..127] */
        HVX_Vector b   = Q6_Vb_vpack_VhVh_sat(h23, h01); /* 128 i8 */

        storev(out + i, b);
    }

    /* Scalar tail */
    for (; i < n; i++) {
        int32_t raw  = (int32_t)table[idx[i]];
        int64_t v    = (int64_t)raw * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift)
                                 : -(((-v) + half) >> shift);
        r += (int64_t)zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
