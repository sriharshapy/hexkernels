Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n);

Fused multiply-add requantize. The v4 per-element bias array is baked to a single
scalar; all parameters are FIXED constants (bake them in; NOT passed):
    BIAS = 50, MULT = 3, SHIFT = 1, ZP = 0
For each i:
    fma  = (int64)a[i] * (int64)b[i] + BIAS
    v    = fma * MULT
    half = (SHIFT > 0) ? (1 << (SHIFT-1)) : 0
    r    = (v >= 0) ? (v + half) >> SHIFT : -(((-v) + half) >> SHIFT)   (round half AWAY from 0)
    r   += ZP
    out[i] = clamp(r, -128, 127)
Inputs a[], b[] are bounded to [-15,15], so every intermediate (a*b, +BIAS, *MULT)
fits in int16.

n is large (a[], b[] each exceed L2), so the task is DDR-bandwidth-bound. To go fast
you should hide DDR latency: stage tiles of a[] and b[] into VTCM with the uDMA
engine and compute on the on-chip copies, double-buffering so the next tile's DMA
overlaps the current tile's compute; DMA the int8 results back to DDR. VTCM is
identity-mapped at 0xd8400000. uDMA: build a static/global Type-0 descriptor
{next, ctrl=len, src, dst}, then Q6_dmstart_A(&desc) / Q6_R_dmwait() (chain two
descriptors via `next` to move both inputs in one dmstart). The descriptor MUST be
static/global (a stack descriptor no-ops at -O2). n is a multiple of 128; handle any
sub-tile remainder.

Because everything fits int16, narrow the int32 inputs to int16 lanes
(Q6_Vh_vpack_VwVw_sat), multiply with Q6_Vh_vmpyi_VhVh, add BIAS, multiply by MULT,
round half-away in halfword lanes (Q6_Vh_vabs_Vh; add half; Q6_Vh_vasr_VhR; restore
sign with Q6_Q_vcmp_gt_VhVh + Q6_V_vmux_QVV), add ZP, then saturating-pack halfword
-> int8 with Q6_Vb_vpack_VhVh_sat (high-lane arg first for natural order). Include
<hexagon_types.h> and <hexagon_protos.h>. Do NOT write main(). Respond with a single
complete C code block and CLOSE the fence with ```.
