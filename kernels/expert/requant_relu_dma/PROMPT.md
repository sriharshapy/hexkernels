Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *a, int8_t *out, int n);

Requantize int32 -> int8 with fused ReLU. The requant parameters are FIXED
constants (bake them in; they are NOT passed):
    MULT = 13, SHIFT = 3, ZP = 0
For each i:
    v    = (int64)a[i] * MULT
    half = (SHIFT > 0) ? (1 << (SHIFT-1)) : 0
    r    = (v >= 0) ? (v + half) >> SHIFT : -(((-v) + half) >> SHIFT)   (round half AWAY from 0)
    r   += ZP
    r    = max(r, ZP)                                  (fused ReLU floor at zp)
    out[i] = clamp(r, -128, 127)

n is large (a[] exceeds L2), so the task is DDR-bandwidth-bound. To go fast you
should hide DDR latency: stage tiles of a[] into VTCM with the uDMA engine and
compute on the on-chip copy, double-buffering so the next tile's DMA overlaps the
current tile's compute; DMA the int8 results back to DDR. VTCM is identity-mapped at
0xd8400000. uDMA: build a static/global Type-0 descriptor {next, ctrl=len, src, dst},
then Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor MUST be static/global (a
stack descriptor no-ops at -O2). n is a multiple of 128; handle any sub-tile remainder.

Inputs are bounded to [-128,127] so a*MULT fits in int32: work in 32-bit word lanes.
Load int32 vectors, multiply by MULT (Q6_Vw_vmpyi_VwRh with mult broadcast into both
halfwords), round half-away via sign-restore (sg = Q6_Vw_vasr_VwR(v,31); |v| =
(v^sg)-sg; shift; restore sign), add ZP, apply ReLU with Q6_Vw_vmax_VwVw against a zp
splat, then saturating-pack int32 -> int16 -> int8 with Q6_Vh_vpack_VwVw_sat and
Q6_Vb_vpack_VhVh_sat (high-lane arg first for natural order). Include
<hexagon_types.h> and <hexagon_protos.h>. Do NOT write main(). Respond with a single
complete C code block and CLOSE the fence with ```.
