/*
 * i8_scan_max — inclusive prefix maximum, HVX Hillis-Steele two-pass.
 *
 * Pass 1 (vectorized): For each 128-byte chunk, compute the local inclusive
 * prefix max using 7 rounds of vmax + vlalign (shift-right-with-fill -128).
 * After round k the window is 2^k, so after 7 rounds every position holds
 * the max of all input bytes from position 0 within the chunk up to it.
 * Then broadcast the carry-in from the previous chunk with a single vmax.
 *
 * Pass 2 (scalar tail): Handle the remaining <128 bytes.
 *
 * vlalign(vB, vA, R) gives: bytes [R..127] of vA | bytes [0..R-1] of vB.
 * To "shift vA right by R and fill the vacated left lanes with -128":
 *   vlalign(vA, vfill, R)   -- where vfill = vsplat(-128)
 * Because the result is: bytes [R..127] of vA in positions [0..127-R],
 * and bytes [0..R-1] of vfill in positions [128-R..127].
 *
 * Wait — that shifts vA LEFT. Let me re-check the semantics.
 * Q6_V_vlalign_VVR(vB, vA, R):
 *   result[0..127] = concat(vA, vB)[R..R+127]
 * So result[j] = vA[j+R] when j+R < 128, else vB[j+R-128].
 * This extracts a window starting R bytes into (vA,vB), i.e. vA shifted
 * LEFT by R bytes with vB filling from the right.
 *
 * We want a RIGHT shift (shift down by R): result[j] = vA[j-R] for j>=R,
 * and result[j] = fill for j<R.
 * That is: result = concat(vA, vfill)[128-R .. 255-R]
 * Using vlalign: Q6_V_vlalign_VVR(vfill, vA, 128-R) — because:
 *   result[j] = concat(vA, vfill)[j + (128-R)]
 *             = vA[j+128-R]  if j+128-R < 128, i.e. j < R  → vA[j+128-R] = vA wrapped? No.
 *
 * Let me reconsider. Actually the simpler way:
 * Q6_V_valign_VVR(vA, vB, R):
 *   result[j] = concat(vB, vA)[j + R]  (vB is high, vA is low)
 *   = vA[j+R] if j+R < 128, else vB[j+R-128]
 *
 * For a RIGHT shift of vA by R lanes with fill on the LEFT:
 * We want result[j] = vA[j-R] for j >= R, and fill for j < R.
 * result = concat(vfill, vA)[R .. R+127]
 * Using Q6_V_valign_VVR(vfill, vA, R):
 *   result[j] = concat(vA, vfill)[j + R]  -- NO, this has vA as LOW, vfill as HIGH
 *   result[j] = vA[j+R] for j+R < 128, else vfill[j+R-128]
 *   This gives a LEFT shift of vA by R! Not what we want.
 *
 * For RIGHT shift by R with fill=-128 on the left:
 * We want: positions [0..R-1] = fill, positions [R..127] = vA[0..127-R]
 * This is: concat(fill, vA)[R..R+127] where fill is the "high" part.
 * Q6_V_valign_VVR(vfill, vA, R) = concat(vA_HIGH=vfill, vA_LOW=vA)[R..R+127] -- wrong order
 *
 * Correct: Q6_V_valign_VVR(vB, vA, R) = concat(vA, vB)[R..R+127]
 *   where vA is the "low" (earlier) part and vB is the "high" (later) part.
 * So for right shift by R: put vA as the high part:
 *   Q6_V_valign_VVR(vA, vfill, R) = concat(vfill, vA)[R..R+127]
 *   result[j] = vfill[j+R] for j+R < 128, else vA[j+R-128]
 *   This gives: result[0..127-R] = vfill[R..127] = fill, result[128-R..127] = vA[0..R-1]
 *   That's a LEFT shift with fill on the right — still wrong.
 *
 * Let me just use vlalign:
 * Q6_V_vlalign_VVR(vB, vA, R) = concat(vA, vB)[R..R+127]
 *   where the "low/first" bytes come from vA and "high/later" from vB.
 * Actually Hexagon docs: vlalign(Vd,Vu,Vv,Rt) = Vd = Vu:Vv >> (Rt*8) i.e.
 * extracts bytes Rt through Rt+127 from the 256-byte concatenation [Vu|Vv].
 * In the 2-arg form Q6_V_vlalign_VVR(vB, vA, R):
 *   result = [vA | vB][R..R+127] = vA bytes starting at position R,
 *   filling from vB when we run out of vA.
 * Hmm, but the ordering: does vA come first (low addr) or vB?
 *
 * The Hexagon ISA: VALIGN(Vd32,Vu32,Vv32,Rt8) -- Vu is the "upper" register,
 * Vv is the "lower" register. Result = byte at position (Rt + j) within [Vv|Vu].
 * So result[j] = Vv[Rt+j] if Rt+j < 128, else Vu[Rt+j-128].
 *
 * Q6_V_valign_VVR(Vu, Vv, R):
 *   result[j] = Vv[R+j] for j < 128-R, else Vu[R+j-128]
 *   = Vv shifted LEFT by R, with Vu filling from the right.
 *
 * For prefix scan right-shift by R (Hillis-Steele):
 * We want: result[j] = vA[j-R] for j >= R, fill for j < R.
 * = vA shifted RIGHT by R, fill on the left.
 * This is: result = [fill | vA] extracted at offset (128-R):
 *   result[j] = fill[128-R+j] for j < R  (since 128-R+j < 128 when j < R)
 *            = vA[128-R+j-128] = vA[j-R] for j >= R
 * Q6_V_valign_VVR(vA, vfill, 128-R):
 *   result[j] = vfill[128-R+j] for j < R → fill ✓
 *   result[j] = vA[128-R+j-128] = vA[j-R] for j >= R ✓
 *
 * So: right_shift_fill(vA, R) = Q6_V_valign_VVR(vA, vfill_neg128, 128 - R)
 * But R must be < 128, and 128-R must be in [0,127] for the immediate form.
 * The immediate R is 0..127 in Hexagon ISA.
 *
 * Actually for round k, R = 1,2,4,8,16,32,64. So 128-R = 127,126,124,120,112,96,64.
 * All are in [1..127], valid.
 *
 * Summary:
 *   shifted = Q6_V_valign_VVR(v, vfill, 128 - shift_amount)
 *   v = Q6_Vb_vmax_VbVb(v, shifted)
 */

#include "kernel_api.h"
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, int8_t *out, int n) {
    const int VLEN = 128;
    /* vfill = splat(-128) = splat(0x80) -- identity for max */
    HVX_Vector vfill = Q6_V_vsplat_R(0x80808080);
    int8_t carry = -128;

    int i = 0;
    for (; i + VLEN <= n; i += VLEN) {
        HVX_Vector v = *(const HVX_Vector *)(in + i);

        /* Hillis-Steele prefix max within the vector.
         * After round k (shift=2^(k-1)): each position p holds max(p-2^k+1 .. p).
         * After 7 rounds: each position holds max(0..p) within the chunk. */
        /* Round 1: shift right by 1 */
        v = Q6_Vb_vmax_VbVb(v, Q6_V_valign_VVR(v, vfill, 127));
        /* Round 2: shift right by 2 */
        v = Q6_Vb_vmax_VbVb(v, Q6_V_valign_VVR(v, vfill, 126));
        /* Round 3: shift right by 4 */
        v = Q6_Vb_vmax_VbVb(v, Q6_V_valign_VVR(v, vfill, 124));
        /* Round 4: shift right by 8 */
        v = Q6_Vb_vmax_VbVb(v, Q6_V_valign_VVR(v, vfill, 120));
        /* Round 5: shift right by 16 */
        v = Q6_Vb_vmax_VbVb(v, Q6_V_valign_VVR(v, vfill, 112));
        /* Round 6: shift right by 32 */
        v = Q6_Vb_vmax_VbVb(v, Q6_V_valign_VVR(v, vfill, 96));
        /* Round 7: shift right by 64 */
        v = Q6_Vb_vmax_VbVb(v, Q6_V_valign_VVR(v, vfill, 64));

        /* Apply carry from previous chunk: splat carry into a vector and vmax */
        HVX_Vector vcarry = Q6_V_vsplat_R((int32_t)(carry & 0xFF) * 0x01010101);
        v = Q6_Vb_vmax_VbVb(v, vcarry);

        /* Store */
        *(HVX_Vector *)(out + i) = v;

        /* Update carry = last byte of this chunk's output */
        carry = out[i + VLEN - 1];
    }

    /* Scalar tail */
    for (; i < n; i++) {
        if (in[i] > carry) carry = in[i];
        out[i] = carry;
    }
}
