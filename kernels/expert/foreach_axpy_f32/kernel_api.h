#ifndef KERNEL_API_H
#define KERNEL_API_H
/*
 * Foreach fp32 AXPY across T tensors each of length L, stored contiguously.
 * Tensor t occupies x[t*L .. t*L+L) (read-only) and y[t*L .. t*L+L) (in-place update).
 * Semantics: y[t*L+i] = alpha * x[t*L+i] + y[t*L+i]  for all t in [0,T), i in [0,L).
 * alpha = 0.5f.  Total elements: T*L = 4*256 = 1024.
 */
void candidate_kernel(const float *x, float *y, int T, int L, float alpha);
#endif
