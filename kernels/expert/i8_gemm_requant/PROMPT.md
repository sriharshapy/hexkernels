Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *A, const int8_t *B, int8_t *out,
 int M, int N, int K,
 int32_t mult, int shift, int8_t zp);
Compute fused int8 linear layer: int8 matmul then requantize to int8.
Step 1 — matmul: acc[i*N+j] = sum_k A[i*K+k]*B[k*N+j] (int32 accumulator)
Step 2 — requantize each acc to int8:
 v = (int64_t)acc * mult
 half = shift > 0 ? (1LL << (shift-1)) : 0
 r = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift) /* round half away from 0 */
 r += zp
 out = saturate_to_int8(r) /* clamp to [-128, 127] */
A is [M x K] (row-major), B is [K x N] (row-major), out is [M x N] int8.
M=32, N=32, K=130 (NOT a multiple of 128) — handle the K-reduction tail.
mult, shift, zp are runtime params — do NOT hardcode them.
Prefer HVX vrmpy for the K-reduction.
Respond with a single complete C code block and CLOSE the fence with ```.
