Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *x, int8_t *out, int n, int alpha, int shift);
Compute int8 leaky ReLU: out[i] = x[i] > 0 ? x[i] : sat8((x[i]*alpha) >> shift), where the
negative-slope result is clamped to [-128,127]. alpha and shift are runtime parameters --
do NOT hardcode them.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast you
should hide DDR latency: stage tiles of x[] into VTCM with the uDMA engine and compute on
the on-chip copy, double-buffering so the next tile's DMA overlaps the current tile's
compute; DMA results back to DDR. VTCM is identity-mapped at 0xd8400000. uDMA: build a
static/global Type-0 descriptor {next,ctrl=len,src,dst}, then `Q6_dmstart_A(&desc)` /
`Q6_R_dmwait()`. The descriptor MUST be static/global (a stack descriptor no-ops at -O2).
n is a multiple of 128; handle any sub-tile remainder.

Idiom: sign-extend bytes to halfwords (`Q6_Wh_vsxt_Vb`), multiply by alpha
(`Q6_Vh_vmpyi_VhVh`), arithmetic-shift + saturating-pack to bytes (`Q6_Vb_vasr_VhVhR_sat`),
the HVX headers. Do NOT write main. Respond with a single complete C code block and
CLOSE the fence with ```.
