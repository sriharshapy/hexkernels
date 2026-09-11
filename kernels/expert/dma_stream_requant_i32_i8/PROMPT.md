Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *a, int8_t *out, int n, int32_t mult, int shift, int8_t zp);
Compute, for i in [0,n):
    out[i] = sat_i8( round_half_away_from_zero(a[i] * mult >> shift) + zp )
using the sign-aware form:
    sm  = a[i] >> 31 (arithmetic)      -- 0 if a[i]>=0, -1 if a[i]<0
    abs = (a[i] ^ sm) - sm             -- |a[i]|
    am  = abs * mult
    half = shift > 0 ? (1 << (shift-1)) : 0
    sh  = (am + half) >> shift (arithmetic)
    r   = (sh ^ sm) - sm + zp
    out[i] = clamp(r, -128, 127)
mult, shift, zp are runtime params (harness fixes mult=200, shift=8, zp=5;
zp is nonzero on purpose). mult fits in 16 bits and is >= 0.

n is large (working set for the int32 input exceeds L2), so the task is
DDR-bandwidth-bound. To go fast, DMA tiles of a[] from DDR into VTCM,
requantize the on-chip copy, and DMA the int8 result tile straight back out,
double-buffering so the next tile's input DMA overlaps the current tile's
requantize+output-DMA. VTCM is identity-mapped at 0xd8400000. uDMA: a
static/global Type-0 descriptor {next, ctrl=len, src, dst}, then
Q6_dmstart_A(&desc)/Q6_R_dmwait(). For the vector path, splat mult into the
low 16 bits of each 32-bit lane and multiply the absolute value with
Q6_Vw_vmpyie_VwVuh (Vd.w[i] = Vu.w[i] * Vv.w[i].uh[0]); pack the requantized
int32 result down to int8 with Q6_Vh_vpack_VwVw_sat then Q6_Vb_vpack_VhVh_sat
(saturating). n is not necessarily a multiple of the DMA tile size; handle
any remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
