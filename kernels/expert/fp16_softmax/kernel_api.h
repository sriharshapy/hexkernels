#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* fp16 row-wise softmax over the LAST axis of an R x C tensor:
 *   rowmax    = max_j x[r][j]
 *   e[r][j]   = exp((float)x[r][j] - rowmax)
 *   rowsum    = sum_j e[r][j]
 *   out[r][j] = (hvx_hf)(e[r][j] / rowsum)
 * R=8 rows, C=256 columns (4 HVX vectors of 64 fp16 lanes per row). Output
 * is compared to a float32 scalar reference with an fp16 tolerance -- HVX
 * float arithmetic is non-IEEE (qf16). HVX has NO vectorized transcendental
 * (exp), so the exp step is necessarily scalar; the max-reduce and the
 * final normalize-by-rowsum ARE vectorized.
 */
void candidate_kernel(const hvx_hf *x, hvx_hf *out, int R, int C);
#endif
