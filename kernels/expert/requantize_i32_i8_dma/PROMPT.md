Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int32_t *a, int8_t *out, int n,
 int32_t mult, int shift, int8_t zp);
For each i in [0,n): v = (int64_t)a[i]*mult; round v half-away-from-zero with
half=(shift>0)?(1<<(shift-1)):0, i.e. r = v>=0 ? (v+half)>>shift : -(((-v)+half)>>shift);
r += zp; out[i] = (int8_t)clamp(r, -128, 127). mult, shift, zp are runtime params --
do NOT hardcode them. Inputs are bounded so a*mult fits int32.

n is large (int32 read + int8 write => DDR-bandwidth-bound). To go fast you should hide
DDR latency: stage tiles of a[] into VTCM with the uDMA engine and compute on the on-chip
copies, double-buffering so the next tile's DMA overlaps the current tile's compute; DMA
results back to DDR. VTCM is identity-mapped at 0xd8400000. uDMA: build a static/global
Type-0 descriptor {next,ctrl=len,src,dst}, then `Q6_dmstart_A(&desc)`/`Q6_R_dmwait()`. The
descriptor MUST be static/global (a stack descriptor no-ops at -O2). Work in word (int32)
lanes; pack 4 word vectors -> 1 byte vector with order-preserving `Q6_Vh_vpacke_VwVw` then `Q6_Vb_vpacke_VhVh`. n is a multiple of 128; handle any sub-tile remainder.

Respond with a single complete C code block and CLOSE the fence with ```.
