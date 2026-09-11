Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *x, int8_t *out, int R, int W,
 const int8_t *gamma, const int8_t *beta,
 const uint8_t *inv_lut);

Row-wise LayerNorm over R rows of width W=128 int8. EACH ROW is normalized independently
using its OWN mean/variance; gamma/beta (per-column, length W, shared across all rows) and
inv_lut (256 entries, shared across all rows) are runtime inputs -- do NOT hardcode them.
Pure fixed-point integer math (no floating point). For each row r in [0, R):
 mu = sum_i(x[r,i]) / W // int32 accumulator, trunc toward zero
 var = sum_i((x[r,i]-mu)^2) / W // int32, trunc, always >= 0
 v_idx = clamp(var, 0, 255)
 inv = inv_lut[v_idx]
 for i in [0, W):
 d = x[r,i] - mu
 scaled = (d*gamma[i] + 64) >> 7 // round-half-up
 normed = (scaled*inv + 128) >> 8 // round-half-up, SHIFT=8
 out[r,i] = clamp(normed + beta[i], -128, 127)
x, out are [R x W] int8 row-major. W is fixed to 128.

R is large and the task is DDR-bandwidth-bound. To go fast, DMA row-blocks of x[] from DDR
into VTCM (double-buffered: prefetch the next block while the current block normalizes),
compute each row's mean/variance/LUT-lookup/affine transform on the on-chip copy, then DMA
the normalized block back to DDR. VTCM is identity-mapped at 0xd8400000. uDMA: static/global
Type-0 descriptor {next,ctrl=len,src,dst}, `Q6_dmstart_A(&desc)`/`Q6_R_dmwait()` (a stack
descriptor no-ops at -O2). Each row is independent -- do NOT mix statistics across rows.

Use HVX intrinsics where helpful (the DMA/VTCM staging is the main speed lever; a scalar
the HVX headers. Do NOT write main. Respond with a single complete C code block and
CLOSE the fence with ```.
