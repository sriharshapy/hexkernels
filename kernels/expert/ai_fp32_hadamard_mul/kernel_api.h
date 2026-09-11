#ifndef KERNEL_API_H
#define KERNEL_API_H
/* FP32 elementwise (Hadamard) product, n=1000.
 * out[i] = a[i] * b[i]  for i in [0, n).
 * Scalar reference: one correctly-rounded IEEE fp32 multiply per element.
 * HVX v68 has no correctly-rounded native sf*sf vector multiply, so an
 * HVX-vectorized candidate is tolerance-close (hvx_close_f32), not
 * necessarily bit-exact.  No reductions.
 */
void candidate_kernel(const float *a, const float *b, float *out, int n);
#endif
