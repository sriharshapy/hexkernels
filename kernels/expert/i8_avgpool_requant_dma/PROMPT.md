Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, int8_t *out, int w, int h,
                          int32_t mult, int shift, int8_t zp);

Fused 2x2 NON-overlapping average pool (signed int8, truncate toward zero) then requantize
to int8 (window/stride are FIXED constants). Output is (w/2) x (h/2), row-major:
  pool = (in[2oy][2ox] + in[2oy][2ox+1] + in[2oy+1][2ox] + in[2oy+1][2ox+1]) / 4
         (int32 accumulator; division truncates TOWARD ZERO)
  v    = (int64_t)pool * mult
  half = shift > 0 ? (1LL << (shift-1)) : 0
  r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)   // round half away from zero
  r   += zp
  out  = saturate_to_int8(r)      // clamp to [-128,127]
mult, shift, zp are runtime params -- do NOT hardcode them (the harness sweeps 2 sets).

The image is large (working set exceeds L2), so this is DDR-bandwidth-bound. To go fast,
DMA row blocks of in[] from DDR into VTCM and pool+requantize the fast on-chip copies,
double-buffering so the next block's DMA overlaps the current block's compute; DMA results
back to DDR. Non-overlapping windows use disjoint input rows, so no halo is needed. VTCM is
identity-mapped at 0xd8400000. uDMA: build a static/global Type-0 descriptor
{next,ctrl=len,src,dst}, then Q6_dmstart_A(&desc)/Q6_R_dmwait() (a stack descriptor no-ops
at -O2). Deinterleave columns (Q6_W_vdeal_VVR), sum the 4 pixels in i16 (sign-extend with
Q6_Wh_vsxt_Vb), truncate /4 toward zero via sign-split, requant, saturating-pack with
Q6_Vb_vasr_VhVhR_sat.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
