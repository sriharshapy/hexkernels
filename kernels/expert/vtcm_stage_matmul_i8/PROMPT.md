Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *X, int32_t *C, int M, int K, int N);

Compute a GEMM: C[m*N+j] = sum_{k=0}^{K-1} A[m*K+k] * X[k*N+j], for m in [0,M),
j in [0,N). A is [M x K] row-major (row m contiguous over k) -- the LARGE
operand, streamed one row at a time. X is [K x N] row-major (row k contiguous
over j) -- the SMALL operand (N<=4), IDENTICAL and reused for every one of the
M rows, so stage/pack it ONCE at kernel entry, not per row. int32 accumulation
is safe: |A[i]|,|X[i]| <= 127, so |sum| <= 127*127*K, comfortably inside int32
range for the given K.

Fixed sizes: M=161, K=8101 (NOT a multiple of 4 -- exercises a vrmpy k-group
tail), N=4. A is ~1.24MB (exceeds L2), so this is DDR-bandwidth-bound on the A
side; X is only ~32KB and constant across rows.

To go fast: pack X ONCE into a vrmpy-friendly k-group layout (4 bytes of X per
output column j, per k-group), then for each of A's M rows, DMA the WHOLE row
(K bytes, one flat transfer) from DDR into a VTCM buffer. Double-buffer across
rows: prefetch row m+1's DMA while computing row m's reduction, hiding DDR
latency behind compute.

Scalar VTCM accesses cost ~48 cycles EACH in timing mode (far worse than a
scalar DDR access) -- so the vrmpy trick's scalar 4-byte splat reads must NEVER
touch the VTCM address directly. Bulk VECTOR-copy (128B at a time) each DMA'd
row out of VTCM into a small cacheable scratch buffer first, and run the
scalar splat extraction against THAT buffer instead. VTCM is identity-mapped
at 0xd8400000. uDMA: build a static/global Type-0 descriptor
{next, ctrl=len, src, dst}, then Q6_dmstart_A(&desc) / Q6_R_dmwait(). The
descriptor MUST be static/global (a stack descriptor no-ops at -O2). K is not
a multiple of 4; handle the k-group tail by zero-padding out-of-range bytes.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT
write main(). Respond with a single complete C code block and CLOSE the fence
with ```.
