#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Narrow table lookup / gather: out[i] = (int8_t)table[idx[i]] for i in
 * [0, n_idx). `table` holds table_words int32 values in the int8 range
 * (-128..127); `idx` holds n_idx indices in [0, table_words); `out` is the
 * narrowed int8 result.
 *
 * table_words is large (table > L1), so a scalar gather misses to DDR on
 * almost every lookup. The achievability bar stages the whole table into
 * VTCM once (uDMA), then uses the HVX hardware word-gather
 * (Q6_vgather_ARMVw) to fetch 32 table words per instruction using
 * runtime index offsets, narrowing each result to int8 on the way out.
 * Requires the VTCM identity translation (add_translation), which the
 * harness sets up. */
void candidate_kernel(const int32_t *table, const int32_t *idx, int8_t *out,
                      int table_words, int n_idx);
#endif
