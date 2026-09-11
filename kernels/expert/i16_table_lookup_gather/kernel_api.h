#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * int16 table lookup / gather:  out[i] = table[idx[i]]  for i in [0, n_idx).
 *   table       : table_hwords int16 values (large, DDR-resident)
 *   idx         : n_idx indices in [0, table_hwords)
 *   out         : n_idx int16 results
 *
 * The Hexagon HVX halfword hardware gather (Q6_vgather_ARMVh) fetches 64
 * halfwords per instruction from a VTCM-resident region using a vector of
 * halfword byte-offsets. Stage the table in VTCM once, then gather. The scalar
 * baseline misses to DDR on every random lookup. The VTCM identity translation
 * is installed by the harness.
 */
void candidate_kernel(const int16_t *table, const int16_t *idx, int16_t *out,
                      int table_hwords, int n_idx);
#endif
