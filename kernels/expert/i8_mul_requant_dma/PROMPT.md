Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n,
                          int32_t mult, int shift, int8_t zp);
For each i in [0,n): prod = a[i]*b[i]; v = prod*mult; round v by half-away-from-zero
with half = (shift>0)?(1<<(shift-1)):0, i.e. r = v>=0 ? (v+half)>>shift :
-(((-v)+half)>>shift); then r += zp; out[i] = saturate_to_int8(r). mult, shift, zp are
runtime parameters -- do NOT hardcode them. Inputs are bounded so prod*mult fits int16.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast you
should hide DDR latency: stage tiles of a[] and b[] into VTCM with the uDMA engine and
compute on the on-chip copies, double-buffering so the next tiles' DMA overlaps the
current tile's compute; DMA results back to DDR. VTCM is identity-mapped at 0xd8400000.
uDMA: build static/global Type-0 descriptors {next,ctrl=len,src,dst}, then
Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor MUST be static/global (a stack
descriptor no-ops at -O2). n is a multiple of 128; handle any sub-tile remainder.
Widen with Q6_Wh_vsxt_Vb, multiply with Q6_Vh_vmpyi_VhVh, pack with Q6_Vb_vasr_VhVhR_sat.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
