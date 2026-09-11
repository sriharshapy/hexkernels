#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Embedding-style D-wide row gather, VTCM-staged:
 *   out[i, :] = table[idx[i], :]     for i in [0, T)
 *
 * Pinned formula (pure gather, no arithmetic):
 *   out[i*E + e] = table[ ((int32_t)idx[i]) * E + e ]     for e in [0, E)
 *
 * table: [vocab_size x E] int8, row-major, 128-byte aligned rows (E is a
 *        multiple of 128 = one HVX vector; table is small enough to fit in
 *        VTCM but larger than L1, so a naive per-row DDR read misses cache
 *        on essentially every random row).
 * idx:   [T] int32, 0 <= idx[i] < vocab_size (guaranteed by the harness).
 * out:   [T x E] int8, row-major.
 *
 * T is large (output working set exceeds L2), so this is a random-row-
 * access / bandwidth-bound gather. The achievability bar DMA-stages the
 * WHOLE table into VTCM once (a single Type-0 uDMA transfer), then does a
 * fast on-chip 128-byte vector copy per row instead of a cold DDR read per
 * row.
 */
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int E, int vocab_size);
#endif
