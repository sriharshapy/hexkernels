#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Embedding gather: out[i, :] = table[idx[i], :]
 *
 * For each token index i in [0, T):
 *   Copy E bytes from table row idx[i] to output row i.
 *   out[i*E + e] = table[idx[i]*E + e]    for e in [0, E)
 *
 * Pinned formula (NO arithmetic — pure gather):
 *   out[i*E + e] = table[ ((int32_t)idx[i]) * E + e ]
 *
 * Constraints:
 *   0 <= idx[i] < VOCAB_SIZE  (guaranteed by harness; no bounds check needed in kernel)
 *   Indices are int32.
 *   T=33 (NOT a multiple of 128), E=128 (embedding dim, a multiple of 128 for HVX).
 *   VOCAB_SIZE=64.
 *
 * table: [VOCAB_SIZE x E] int8, row-major, aligned.
 * idx:   [T] int32.
 * out:   [T x E] int8, row-major, aligned.
 */
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int E);
#endif /* KERNEL_API_H */
