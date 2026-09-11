Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int8_t *out, int n,
                          int32_t scale, int32_t shift, int s);
Compute out[i] = sat8( round_half_away_from_zero(a[i]*scale + shift) >> s ), i.e. with
v = a[i]*scale + shift and half = (s>0 ? 1<<(s-1) : 0),
r = (v >= 0) ? (v + half) >> s : -(((-v) + half) >> s), clamped to [-128,127].
scale, shift, s are runtime parameters -- do NOT hardcode them.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast you
should hide DDR latency: stage tiles of a[] into VTCM with the uDMA engine and compute on
the on-chip copy, double-buffering so the next tile's DMA overlaps the current tile's
compute; DMA results back to DDR. VTCM is identity-mapped at 0xd8400000. uDMA: build a
static/global Type-0 descriptor {next,ctrl=len,src,dst}, then Q6_dmstart_A(&desc) /
Q6_R_dmwait(). The descriptor MUST be static/global (a stack descriptor no-ops at -O2).
n is a multiple of 128; handle any sub-tile remainder.

For the chosen params, a*scale+shift fits int16 so you can work in halfword lanes:
sign-extend (Q6_Wh_vsxt_Vb), multiply-add, take |v| (Q6_Vh_vabs_Vh) for round-away,
add half, arithmetic-shift, restore sign, then saturating-pack to bytes
(Q6_Vb_vasr_VhVhR_sat with shift 0). Include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT write main(). Respond with a single complete C code block and CLOSE the fence with ```.
