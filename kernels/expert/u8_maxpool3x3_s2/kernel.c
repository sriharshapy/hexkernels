/*
 * u8_maxpool3x3_s2 HVX kernel — 3x3 max pool stride 2, clamp-to-edge.
 * Pinned shapes: W=129, H=97, OW=65, OH=49.
 *
 * Strategy:
 *   For each output row oy, compute hmaxrow[0..64] for 3 clamped input rows,
 *   then vertical vmax and write 65 bytes.
 *
 * hmax_row approach (avoids vdeal):
 *   Load A = r[0..127] aligned.
 *   Load B = r[1..128] via loadu (r[128] exists; W=129).
 *   After Q6_Vb_vdeal_Vb(A):
 *     even64[k] = A[2k] = r[2k]    for k=0..63  <- center column
 *     odd64[k]  = A[2k+1] = r[2k+1] for k=0..63 <- right column (in positions 64..127 of dealt)
 *
 *   Similarly Q6_Vb_vdeal_Vb(B):
 *     even64_B[k] = B[2k] = r[2k+1] for k=0..63  <- same as odd of A, but 0-indexed differently
 *     odd64_B[k]  = B[2k+1] = r[2k+2] for k=0..63
 *
 *   Wait -- B[k] = r[k+1], so:
 *     Q6_Vb_vdeal_Vb(B): even_B[k] = B[2k] = r[2k+1] (same as odd of A)
 *                         odd_B[k]  = B[2k+1] = r[2k+2]
 *
 *   hmax at output position k (center = r[2k]):
 *     left  = r[2k-1] = A[2k-1] = odd_A[k-1]  for k>=1, r[0] for k=0
 *     center = r[2k] = A[2k] = even_A[k]
 *     right  = r[2k+1] = A[2k+1] = odd_A[k]
 *
 *   hmax[k] = max(left[k], even_A[k], odd_A[k])
 *
 *   For the left column, we use a simple trick: instead of shifting odd_A, note:
 *     left[k] = odd_A[k-1] = even_B[k-1]  (since even_B[k]=r[2k+1]=odd_A[k])
 *
 *   Hmm this is getting complex. Let me use an alternative method:
 *
 * CLEANER METHOD using Q6_W_vdeal_VVR (2-vector form):
 *   Q6_W_vdeal_VVR(Va, Vb, -1): deinterleaves the PAIR (Vb is low, Va is high):
 *     lo[k] = Vb[2k] for k=0..63
 *     hi[k] = Vb[2k+1] for k=0..63
 *   ... actually I'm not sure about this.
 *
 * SIMPLEST CORRECT: use the 2-vector form to get even and odd in separate vectors.
 *
 * Actually: use the approach from k03 directly:
 *   dealt = Q6_Vb_vdeal_Vb(A)
 *   shifted = Q6_V_valign_VVR(dealt, dealt, 64)
 *   -- but here shifted[0..63] = dealt[64..127] = odd_A bytes
 *   right_col = shifted  (the 64 odd bytes, now in positions 0..63)
 *   center_col = dealt   (even bytes in positions 0..63)
 *   left_col = shift right_col right by 1, fix pos 0 with center_col[0]
 *
 *   For left_col shift: Q6_V_valign_VVR(right_col, vzero, 127):
 *     result[0] = vzero[127] = 0
 *     result[k] = right_col[k-1] = odd_A[k-1] for k=1..63
 *   Then fix position 0 with center_col[0].
 *
 *   BUT WAIT: Q6_V_valign_VVR shifts the FIRST ARGUMENT right, using SECOND as fill.
 *   "right-shift right_col by 1 byte, fill left with vzero[127]=0":
 *   Q6_V_valign_VVR(right_col, vzero, 127):
 *     result = concat(vzero, right_col)[127..254]
 *     result[0] = vzero[127] = 0
 *     result[k] = right_col[k-1] for k=1..127
 *   result[0..63]: 0, right_col[0], right_col[1], ..., right_col[62] ✓
 *
 *   Fix position 0: mux with center_col using q1 = Q6_Q_vsetq2_R(1):
 *   left_col = Q6_V_vmux_QVV(q1, center_col, left_shifted)
 *   = center_col[0] at pos 0, left_shifted[k] at pos k>0
 *
 *   hmax[k] = max(left_col[k], center_col[k], right_col[k]) for k=0..63
 *
 * This approach computes only the 64 interior output pixels. The 65th (ox=64) is done
 * scalarly: max(r[127], r[128]).
 *
 * Note on right_col positions 64..127: those are dealt[64+64..127] = dealt[128..191] = wrapped.
 *   Q6_V_valign_VVR(dealt, dealt, 64): result[k] = dealt[(k+64)%128]
 *   result[64..127] = dealt[0..63] = even bytes
 * But we only use right_col[0..63], so positions 64..127 are irrelevant.
 *
 * HVX idioms: Q6_Vb_vdeal_Vb, Q6_V_valign_VVR, Q6_Vub_vmax_VubVub,
 *             Q6_V_vmux_QVV, Q6_Q_vsetq2_R.
 */

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>

#define W  129
#define H   97
#define OW  65
#define OH  49

static inline HVX_Vector loadu128(const void *p)
{
    HVX_Vector v;
    __builtin_memcpy(&v, p, 128);
    return v;
}

/*
 * Compute hmaxrow[0..63] (64 values) for row r[0..128].
 * Also returns hmax64 = max(r[127], r[128]) for ox=64.
 * Out64 is a 128-byte aligned buffer; only bytes 0..63 are set (ox=0..63).
 */
static inline void hmax_row64(const uint8_t * restrict r,
                               uint8_t * restrict out64,
                               uint8_t *hmax64)
{
    HVX_Vector A = loadu128(r);  /* r[0..127]; r may not be 128B-aligned (W=129, iy*W may not be mult of 128) */

    /* Deinterleave A: dealt[0..63]=r[0,2,...,126], dealt[64..127]=r[1,3,...,127] */
    HVX_Vector dealt = Q6_Vb_vdeal_Vb(A);

    /* center_col[k] = r[2k] for k=0..63  (positions 0..63 of dealt) */
    HVX_Vector center_col = dealt;

    /* right_col[k] = r[2k+1] for k=0..63 (shift odd bytes to positions 0..63)
     * Q6_V_valign_VVR(dealt, dealt, 64): result[k] = dealt[k+64] for k+64<128, else dealt[k-64]
     * result[0..63] = dealt[64..127] = r[1,3,...,127] ✓ */
    HVX_Vector right_col = Q6_V_valign_VVR(dealt, dealt, 64);

    /* left_col[k] = r[2k-1] for k>=1, r[0] for k=0
     * = right-shift right_col by 1, fix position 0 with center_col[0].
     * Q6_V_valign_VVR(right_col, vzero, 127): result[0]=0, result[k]=right_col[k-1] for k=1..63
     * Then mux: left_col[0] = center_col[0] = r[0] (clamped border), left_col[k]=right_col[k-1] */
    HVX_Vector vzero      = Q6_V_vzero();
    HVX_Vector left_shifted = Q6_V_valign_VVR(right_col, vzero, 127);
    HVX_VectorPred q1       = Q6_Q_vsetq2_R(1);
    HVX_Vector left_col   = Q6_V_vmux_QVV(q1, center_col, left_shifted);

    /* hmax[k] = max(left_col[k], center_col[k], right_col[k]) for k=0..63 */
    HVX_Vector hmax_row_vec = Q6_Vub_vmax_VubVub(left_col,
                                  Q6_Vub_vmax_VubVub(center_col, right_col));

    *(HVX_Vector *)out64 = hmax_row_vec;

    /* Scalar ox=64: max(r[127], r[128]) */
    uint8_t r127 = r[127];
    uint8_t r128 = r[128];  /* valid since W=129 */
    *hmax64 = (r127 > r128) ? r127 : r128;
}

void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h)
{
    (void)w; (void)h;

    /* Aligned scratch for 3 rows of 65 hmax values.
     * hrow[0..63] = HVX vector result, hrow[64] = scalar last pixel. */
    uint8_t hrow0[128] __attribute__((aligned(128)));
    uint8_t hrow1[128] __attribute__((aligned(128)));
    uint8_t hrow2[128] __attribute__((aligned(128)));
    uint8_t h64_0, h64_1, h64_2;

    for (int oy = 0; oy < OH; oy++) {
        int cy  = 2 * oy;
        int iy0 = (cy > 0)     ? cy - 1 : 0;
        int iy1 = cy;
        int iy2 = (cy < H - 1) ? cy + 1 : H - 1;

        hmax_row64(in + iy0 * W, hrow0, &h64_0);
        hmax_row64(in + iy1 * W, hrow1, &h64_1);
        hmax_row64(in + iy2 * W, hrow2, &h64_2);

        /* Vertical max of 3 rows (64 bytes via HVX + 1 scalar). */
        HVX_Vector v0  = *(const HVX_Vector *)hrow0;
        HVX_Vector v1  = *(const HVX_Vector *)hrow1;
        HVX_Vector v2  = *(const HVX_Vector *)hrow2;
        HVX_Vector res = Q6_Vub_vmax_VubVub(v0, Q6_Vub_vmax_VubVub(v1, v2));

        __builtin_memcpy(out + oy * OW, &res, OW - 1);  /* 64 bytes (ox=0..63) */

        /* Last pixel ox=64. */
        uint8_t last = h64_0;
        if (h64_1 > last) last = h64_1;
        if (h64_2 > last) last = h64_2;
        out[oy * OW + 64] = last;
    }
}
