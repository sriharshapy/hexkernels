#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Word table lookup / gather:  out[i] = table[idx[i]]  for i in [0, n_idx).
 *   table    : table_words int32 values (large, DDR-resident)
 *   idx      : n_idx indices in [0, table_words)
 *   out      : n_idx int32 results
 *
 * Random indices into a table larger than L1 make a scalar gather miss to DDR
 * on every lookup. The Hexagon HVX hardware gather (Q6_vgather_ARMVw) fetches 32
 * elements per instruction from a VTCM-resident region: stage the table in VTCM
 * once, then gather. Requires the VTCM identity translation (add_translation),
 * which the harness sets up.
 */
void candidate_kernel(const int32_t *table, const int32_t *idx, int32_t *out,
                      int table_words, int n_idx);
#endif
