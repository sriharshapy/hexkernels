#ifndef KERNEL_API_H
#define KERNEL_API_H
/* MSE (Mean Squared Error) loss reduction, n=512.
 * Semantics: out[0] = (1.0f/n) * sum_i( (pred[i] - tgt[i])^2 )
 * Scalar output (reduction). Tolerance compare required (fp32 qfloat path).
 */
void candidate_kernel(const float *pred, const float *tgt, float *out, int n);
#endif
