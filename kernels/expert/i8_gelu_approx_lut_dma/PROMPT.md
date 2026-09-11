Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *in, int8_t *out, int n,
 const int8_t *lut, int8_t scale);
For each i in [0,n): idx = clamp((int)in[i]*scale/64 + 128, 0, 255) where /64 is C
integer division (TRUNCATES toward zero); out[i] = lut[idx]. `scale` (int8 in [1,127])
and the 256-entry `lut` are runtime parameters -- do NOT hardcode them.

n is large (int8 read + int8 write => DDR-bandwidth-bound). To go fast you should hide
DDR latency: stage tiles of in[] into VTCM with the uDMA engine and compute on the on-chip
copies, double-buffering so the next tile's DMA overlaps the current tile's compute; DMA
results back to DDR. VTCM is identity-mapped at 0xd8400000. uDMA: build a static/global
Type-0 descriptor {next,ctrl=len,src,dst}, then `Q6_dmstart_A(&desc)`/`Q6_R_dmwait()`. The
descriptor MUST be static/global (a stack descriptor no-ops at -O2). Compute the index in
halfword lanes (x*scale fits int16; /64 truncation toward zero via sign-split), pack to a
byte index, then look up with the HVX 128B vlut32 family (`Q6_Vb_vlut32_VbVbR` / `Q6_Vb_vlut32or_VbVbVbR`, R=0..7) over two vlut-ready table vectors. n is a multiple of 128;
handle any sub-tile remainder.

Respond with a single complete C code block and CLOSE the fence with ```.
