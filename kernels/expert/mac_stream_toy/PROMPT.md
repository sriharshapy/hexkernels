Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const uint8_t *a, const int8_t *b, int32_t *out, int n, int blk);
n = nblocks*blk. For each block m in [0, n/blk):
 out[m] = sum_{k=0}^{blk-1} (int32)a[m*blk+k] * (int32)b[m*blk+k]
a is uint8 (activations), b is int8 (weights); accumulate in int32 (exact, no saturation).

n is large (working set exceeds L2) so this is bandwidth-bound. Stream it: DMA the a/b tile
for the next block into VTCM (async) while the vrmpy accumulator processes the current block
in VTCM, then horizontally reduce the 32 word lanes and emit out[m]. VTCM is identity-mapped
at 0xd8400000. uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst},
`Q6_dmstart_A(&desc)`/`Q6_R_dmwait()` (descriptor MUST be static/global). Use
`Q6_Vw_vrmpyacc_VwVubVb` (uint8 x int8 -> int32, 4-at-a-time). blk is a multiple of 128.

Respond with a single complete C code block and CLOSE the fence with ```.
