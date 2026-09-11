/*
 * u8_maxpool2x2_pad HVX kernel — 2x2 max pool stride 2, zero-padding.
 * Pinned shapes: W=129, H=97, OW=65, OH=49.
 *
 * Strategy:
 *   ow_full=64, oh_full=48. The main loop processes 64 interior output
 *   columns at once using Q6_Vb_vdeal_Vb to separate even/odd input columns.
 *   Two source rows (row0, row1) are each dealt; horizontal vmax collapses
 *   2 cols → 1; vertical vmax across the two rows → 64 output bytes.
 *   The tail column (ox=64) and last output row (oy=48) are scalar.
 *
 * Unaligned loads: rows start at offset iy*129 which is not 128-aligned
 * (129 % 128 = 1). Use __builtin_memcpy to load 128 bytes safely.
 *
 * Deinterleave trick (Q6_Vb_vdeal_Vb):
 *   Input A[0..127] = r[0,1,2,...,127]
 *   dealt[0..63]  = r[0,2,4,...,126]  (even = "left" column)
 *   dealt[64..127]= r[1,3,5,...,127]  (odd = "right" column)
 *   Q6_V_valign_VVR(dealt, dealt, 64):
 *     result[k] = dealt[(k+64) % 128]
 *     result[0..63] = dealt[64..127] = r[1,3,...,127]  (right col in lo half)
 *   hmax[k] = vmax(dealt[k], result[k]) = max(r[2k], r[2k+1])  for k=0..63
 */

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

#define W   129
#define H    97
#define OW   65
#define OH   49

static inline HVX_Vector loadu128(const void *p)
{
    HVX_Vector v;
    __builtin_memcpy(&v, p, 128);
    return v;
}

/* hmax_row64: given a row pointer, compute max(row[2k], row[2k+1]) for k=0..63.
 * Returns 128-byte vector; bytes [0..63] hold the 64 output values. */
static inline HVX_Vector hmax_row64(const uint8_t * restrict row)
{
    HVX_Vector A    = loadu128(row);             /* row[0..127] */
    HVX_Vector dealt = Q6_Vb_vdeal_Vb(A);        /* [0..63]=row[0,2,...,126]; [64..127]=row[1,3,...,127] */
    HVX_Vector right = Q6_V_valign_VVR(dealt, dealt, 64); /* [0..63]=row[1,3,...,127] */
    return Q6_Vub_vmax_VubVub(dealt, right);      /* [0..63]=max(row[2k],row[2k+1]) */
}

void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h)
{
    (void)w; (void)h;

    /* Interior output rows: oy=0..47 (both input rows iy0=2*oy, iy1=2*oy+1 in bounds) */
    for (int oy = 0; oy < 48; oy++) {
        const uint8_t *row0 = in + (2 * oy)     * W;
        const uint8_t *row1 = in + (2 * oy + 1) * W;
        uint8_t *orow = out + oy * OW;

        /* Horizontal max for each row → 64 output values in bytes [0..63] */
        HVX_Vector hm0 = hmax_row64(row0);
        HVX_Vector hm1 = hmax_row64(row1);

        /* Vertical max across the two rows */
        HVX_Vector res = Q6_Vub_vmax_VubVub(hm0, hm1);

        /* Store 64 bytes (ox=0..63) */
        __builtin_memcpy(orow, &res, 64);

        /* Tail pixel ox=64: c=128, c+1=129 (OOB → 0)
         * max(row0[128], 0, row1[128], 0) = max(row0[128], row1[128]) */
        uint8_t p0 = row0[128];
        uint8_t p1 = row1[128];
        orow[64] = p0 > p1 ? p0 : p1;
    }

    /* Last output row oy=48: iy0=96 (valid, H=97), iy1=97 (OOB → 0) */
    {
        const uint8_t *row0 = in + 96 * W;
        uint8_t *orow = out + 48 * OW;

        /* Horizontal max of last input row (row1 is zero → vmax(hm0, 0) = hm0) */
        HVX_Vector hm0 = hmax_row64(row0);

        /* Store 64 bytes (ox=0..63) */
        __builtin_memcpy(orow, &hm0, 64);

        /* Tail pixel ox=64: c=128, c+1=OOB → max(row0[128], 0) = row0[128] */
        orow[64] = row0[128];
    }
}
