Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *x, int8_t *out,
 int n_ch, int n_elem,
 const int8_t *alpha, int shift);
Per-channel PReLU on int8 data laid out as x[c*n_elem + i] (n_ch channels,
n_elem elements per channel, n_elem a multiple of 128). For each element:
 out = x > 0 ? x : sat8( (x * alpha[c]) >> shift )
where >> is an arithmetic (sign-preserving, floor) shift and sat8 clamps to
[-128,127].

N is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go
fast, DMA channel-aligned tiles of x[] from DDR into VTCM, compute on the fast
on-chip copies, and DMA results back, double-buffering so the next tile's DMA
overlaps the current tile's compute. VTCM is identity-mapped at 0xd8400000.
uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst}, `Q6_dmstart_A(&d)`/
`Q6_R_dmwait()`. Vectorize in halfword lanes (sign-extend bytes, multiply by the
alpha splat, arithmetic-shift, mux the passthrough on x>0, saturating-pack back
to int8).

write main. Respond with a single complete C code block and CLOSE the fence.
