#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Foreach int8 add across T tensors each of length L, stored contiguously.
 * Tensor t occupies a[t*L .. t*L+L) and b[t*L .. t*L+L).
 * Semantics: out[t*L+i] = (int8_t)((int)a[t*L+i] + (int)b[t*L+i])  -- two's-complement wrap.
 * Total elements: T*L = 4*256 = 1024.
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int T, int L);
#endif
