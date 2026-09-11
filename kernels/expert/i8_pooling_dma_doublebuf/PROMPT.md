Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *x, int8_t *out, int n, int window, int shift);
Compute a 1D OVERLAPPING "energy" pool (windowed sum-of-squares), window=8, stride=4 (50%
overlap -- each input byte contributes to TWO consecutive output windows), then requantize:
 for j in [0, n):
 acc = sum over k=0..7 of x[j*4+k]*x[j*4+k] (int32 accumulator)
 out[j] = (int8_t) min(127, acc >> shift)
acc is always >= 0 (sum of squares), so no lower clamp is needed -- only clamp the upper
bound at 127. window is fixed to 8, stride fixed to 4. x has 4*n+4 samples (halo=4); out has
n int8 results. n is a multiple of 128. shift is a runtime int -- do NOT hardcode it.

n is large and each input byte is re-read by two overlapping windows, so this is DDR-
bandwidth-bound with genuine re-read pressure. To go fast, tile the input: DMA a block of
x[] (with a 4-byte halo) from DDR into VTCM, pool+shift+clamp the on-chip copy writing int8
results into a VTCM output slot, then DMA the output slot back to DDR -- double-buffered so
the next input block's DMA overlaps the current block's compute; this also avoids re-
fetching the overlapped halo bytes from DDR on every window. VTCM is identity-mapped at
0xd8400000. uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst},
`Q6_dmstart_A(&desc)`/`Q6_R_dmwait()` (a stack descriptor no-ops at -O2).

Vectorize with the HVX byte reduce-multiply-accumulate intrinsic: for an ALIGNED 128-byte
block, `Q6_Vw_vrmpy_VbVb(v, v)` gives 32 lanes, each the sum of 4 squared bytes (one stride-4
group). Load a SECOND window shifted by 4 bytes (unaligned load via valign) and vrmpy that
too -- adding the two 32-lane results gives exactly the window-8 energy for 32 consecutive
outputs in one shot (each lane needs its own group's sum-of-4 plus the NEXT group's
sum-of-4). Shift, clamp, and copy to the output tile.

Respond with a single complete C code block and CLOSE the fence with ```.
