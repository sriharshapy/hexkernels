#ifndef KERNEL_API_H
#define KERNEL_API_H
/* FP16 elementwise (Hadamard) product, n=1000.
 * out[i] = (hvx_hf)((float)a[i] * (float)b[i]) for i in [0, n).
 * Bit-exact to the scalar reference: one correctly-rounded IEEE multiply per element.
 */
typedef __fp16 hvx_hf;
void candidate_kernel(const hvx_hf *a, const hvx_hf *b, hvx_hf *out, int n);
#endif
