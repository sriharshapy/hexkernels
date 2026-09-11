Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, int8_t *out, int n, int8_t scale);
Compute int8 ReLU6: out[i] = clamp(x[i], 0, cap) where cap = 6 * scale (i.e.
out[i] = x[i] < 0 ? 0 : (x[i] > cap ? cap : x[i])). scale is a runtime parameter --
do NOT hardcode it.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast you
should hide DDR latency: stage tiles of x[] into VTCM with the uDMA engine and compute on
the on-chip copy, double-buffering so the next tile's DMA overlaps the current tile's
compute; DMA results back to DDR. VTCM is identity-mapped at 0xd8400000. uDMA: build a
static/global Type-0 descriptor {next,ctrl=len,src,dst}, then Q6_dmstart_A(&desc) /
Q6_R_dmwait(). The descriptor MUST be static/global (a stack descriptor no-ops at -O2).
n is a multiple of 128; handle any sub-tile remainder.

Idiom: relu6 = Q6_Vb_vmin_VbVb(Q6_Vb_vmax_VbVb(x, zero), cap) with cap broadcast to all
byte lanes. Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main(). Respond
with a single complete C code block and CLOSE the fence with ```.
