Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n);

Residual-add requantize, int32 + int32 -> int8. The requant parameters are FIXED
constants (bake them in; they are NOT passed):
    MULT = 5, SHIFT = 3, ZP = 0
For each i, with sum = (int64)a[i] + (int64)b[i]:
    v    = sum * MULT
    half = (SHIFT > 0) ? (1 << (SHIFT-1)) : 0
    r    = (v >= 0) ? (v + half) >> SHIFT : -(((-v) + half) >> SHIFT)   (round half AWAY from 0)
    r   += ZP
    out[i] = clamp(r, -128, 127)

n is large (a[], b[] each exceed L2), so the task is DDR-bandwidth-bound. To go fast
you should hide DDR latency: stage tiles of a[] and b[] into VTCM with the uDMA
engine and compute on the on-chip copies, double-buffering so the next tile's DMA
overlaps the current tile's compute; DMA the int8 results back to DDR. VTCM is
identity-mapped at 0xd8400000. uDMA: build a static/global Type-0 descriptor
{next, ctrl=len, src, dst}, then Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor
MUST be static/global (a stack descriptor no-ops at -O2). n is a multiple of 128;
handle any sub-tile remainder.

Inputs are bounded to [-128,127] so sum*MULT fits comfortably in int32: you can work
in 32-bit word lanes. Load int32 vectors, add, multiply by MULT (Q6_Vw_vmpyi_VwRh
with the mult broadcast into both halfwords), round half-away via sign-restore
(sg = Q6_Vw_vasr_VwR(v,31); |v| = (v^sg)-sg; shift; restore sign), add ZP, then
saturating-pack int32 -> int16 -> int8 with Q6_Vh_vpack_VwVw_sat and
Q6_Vb_vpack_VhVh_sat (high-lane arg first for natural order). Include
<hexagon_types.h> and <hexagon_protos.h>. Do NOT write main(). Respond with a single
complete C code block and CLOSE the fence with ```.
