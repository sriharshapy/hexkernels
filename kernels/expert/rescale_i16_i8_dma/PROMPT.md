Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int16_t *a, int8_t *out, int n);

Rescale int16 -> int8. The parameters are FIXED constants (bake them in; NOT passed):
    MULT = 3, SHIFT = 4, ZP = 0
For each i:
    v    = (int32)a[i] * MULT
    half = (SHIFT > 0) ? (1 << (SHIFT-1)) : 0
    r    = (v >= 0) ? (v + half) >> SHIFT : -(((-v) + half) >> SHIFT)   (round half AWAY from 0)
    r   += ZP
    out[i] = clamp(r, -128, 127)

n is large (a[] exceeds L2), so the task is DDR-bandwidth-bound. To go fast you
should hide DDR latency: stage tiles of a[] into VTCM with the uDMA engine and
compute on the on-chip copy, double-buffering so the next tile's DMA overlaps the
current tile's compute; DMA the int8 results back to DDR. VTCM is identity-mapped at
0xd8400000. uDMA: build a static/global Type-0 descriptor {next, ctrl=len, src, dst},
then Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor MUST be static/global (a
stack descriptor no-ops at -O2). n is a multiple of 128; handle any sub-tile remainder.

a[] spans the full int16 range, so a*MULT can exceed int16: sign-extend the int16
inputs to int32 word lanes (Q6_Ww_vsxt_Vh), multiply by MULT (Q6_Vw_vmpyi_VwRh, mult
broadcast into both halfwords), round half-away via sign-restore (sg =
Q6_Vw_vasr_VwR(v,31); |v| = (v^sg)-sg; shift; restore sign), add ZP, then
saturating-pack int32 -> int16 -> int8 with Q6_Vh_vpack_VwVw_sat and
Q6_Vb_vpack_VhVh_sat (high-lane arg first for natural order). Include
<hexagon_types.h> and <hexagon_protos.h>. Do NOT write main(). Respond with a single
complete C code block and CLOSE the fence with ```.
