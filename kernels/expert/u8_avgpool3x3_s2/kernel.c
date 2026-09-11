/*
 * u8_avgpool3x3_s2 HVX kernel — 3x3 avg pool stride 2, clamp-to-edge, truncating divide.
 * Pinned shapes: W=129, H=97, OW=65, OH=49.
 *
 * Strategy:
 *   For each output row oy, copy 3 clamped input rows to aligned 130-byte buffers,
 *   compute horizontal 3-tap sums (left+center+right) for ox=0..63 via HVX,
 *   accumulate 3 row sums vertically, then divide by 9 via scalar loop through tot_buf.
 *   ox=64 (last output pixel) is handled entirely scalarly.
 *
 * HVX idioms: Q6_Vb_vdeal_Vb, Q6_V_vror_VR, Q6_V_valign_VVR, Q6_V_vmux_QVV,
 *             Q6_Wuh_vunpack_Vub, Q6_Vh_vadd_VhVh, Q6_Q_vsetq2_R.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define W   129
#define H    97
#define OW   65
#define OH   49

/* Copy W=129 bytes from src to 128-byte aligned dst, appending src[W-1] at dst[W].
 * This ensures dst is aligned and dst[0..128] covers all needed input bytes. */
static inline void copy_row(uint8_t *dst, const uint8_t *src)
{
    for (int i = 0; i < W; i++) dst[i] = src[i];
    dst[W] = src[W-1];
}

/*
 * Compute horizontal 3-tap sums for output columns ox=0..63.
 * Input row r[0..128] is in an aligned 130-byte buffer.
 * center[k] = r[2k], left[k] = r[max(2k-1,0)], right[k] = r[2k+1].
 * Returns a vector of 64 u16 values (in positions 0..63, bytes 0..127).
 * Also writes s_last = r[127] + r[128] + r[128] (hsum for ox=64, right-clamped).
 */
static inline HVX_Vector row_hsum3(const uint8_t *row, uint16_t *s_last)
{
    /* Load aligned 128 bytes: row[0..127] */
    HVX_Vector A = *(const HVX_Vector *)row;

    /* Deinterleave: dealed[k] = row[2k] for k=0..63 (even bytes, i.e., center columns)
     *               dealed[64+k] = row[2k+1] for k=0..63 (odd bytes, i.e., right columns) */
    HVX_Vector dealed = Q6_Vb_vdeal_Vb(A);

    /* Mask to keep only the low 64 bytes */
    HVX_VectorPred m64 = Q6_Q_vsetq2_R(64);

    /* even64[k] = row[2k] for k=0..63 (center col), zeros for k=64..127 */
    HVX_Vector even64 = Q6_V_vand_QV(m64, dealed);

    /* Rotate dealed right by 64: brings odd bytes from positions 64..127 to 0..63 */
    HVX_Vector rot64 = Q6_V_vror_VR(dealed, 64);

    /* odd64[k] = row[2k+1] for k=0..63 (right col), zeros for k=64..127 */
    HVX_Vector odd64 = Q6_V_vand_QV(m64, rot64);

    /* Build left col: left[k] = right[k-1] for k>=1, left[0] = center[0] (clamp-to-edge).
     * Q6_V_valign_VVR(odd64, vzero, 127): result[0]=0, result[k]=odd64[k-1] for k>=1.
     * Then mux position 0: use even64[0] = row[0] as the clamped left border. */
    HVX_Vector vzero   = Q6_V_vzero();
    HVX_Vector shifted = Q6_V_valign_VVR(odd64, vzero, 127);
    HVX_VectorPred q1  = Q6_Q_vsetq2_R(1);
    HVX_Vector left64  = Q6_V_vmux_QVV(q1, even64, shifted);

    /* Unpack each column's 64 u8 values to 64 u16 values (lo half of pair) */
    HVX_Vector uc = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(even64));
    HVX_Vector ul = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(left64));
    HVX_Vector ur = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(odd64));

    /* Horizontal sum: hsum[k] = left[k] + center[k] + right[k] as u16 */
    HVX_Vector hsum = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(uc, ul), ur);

    /* Scalar hsum for ox=64: center=row[128], left=row[127], right=row[128] (clamped) */
    *s_last = (uint16_t)((uint32_t)row[127] + row[128] + row[128]);

    return hsum;
}

void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h)
{
    (void)w; (void)h;

    /* Aligned row buffers: 256 bytes each to guarantee W+1=130 bytes fit aligned */
    uint8_t rb0[256] __attribute__((aligned(128)));
    uint8_t rb1[256] __attribute__((aligned(128)));
    uint8_t rb2[256] __attribute__((aligned(128)));

    /* 64 u16 total sums for divide-by-9 via scalar */
    uint16_t tot_buf[64] __attribute__((aligned(128)));

    for (int oy = 0; oy < OH; oy++) {
        int cy  = 2 * oy;
        int iy0 = (cy > 0)   ? cy - 1 : 0;
        int iy1 = cy;
        int iy2 = (cy < H-1) ? cy + 1 : H - 1;

        /* Copy clamped rows to aligned buffers */
        copy_row(rb0, in + iy0 * W);
        copy_row(rb1, in + iy1 * W);
        copy_row(rb2, in + iy2 * W);

        /* Horizontal sums for each row */
        uint16_t s0, s1, s2;
        HVX_Vector h0 = row_hsum3(rb0, &s0);
        HVX_Vector h1 = row_hsum3(rb1, &s1);
        HVX_Vector h2 = row_hsum3(rb2, &s2);

        /* Vertical sum: tot[k] = h0[k] + h1[k] + h2[k] (9-tap total, u16) */
        HVX_Vector tot = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(h0, h1), h2);

        /* Store tot to aligned buffer for scalar divide */
        *(HVX_Vector *)tot_buf = tot;

        /* Divide by 9 (integer truncation) and write output ox=0..63 */
        uint8_t *orow = out + oy * OW;
        for (int ox = 0; ox < 64; ox++)
            orow[ox] = (uint8_t)(tot_buf[ox] / 9);

        /* ox=64: scalar path */
        orow[64] = (uint8_t)((uint32_t)(s0 + s1 + s2) / 9);
    }
}
