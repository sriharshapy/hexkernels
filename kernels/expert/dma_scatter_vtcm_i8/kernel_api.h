#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Scatter: out[idx[i]] = (int32_t)values[i] for i in [0, n). `idx` is a
 * PERMUTATION of [0, n) (each destination position is written exactly
 * once, so the result is independent of any internal lane-processing
 * order -- this is what makes the operation well-defined for a
 * vector/hardware scatter). `values` holds int8-range data; `out` holds
 * n int32 slots (word-granularity, since the HVX hardware scatter writes
 * whole words/halfwords, not individual bytes).
 *
 * The achievability bar scatters directly into a VTCM-resident output
 * table (Q6_vscatter_RMVwV), then DMAs the completed table out to DDR in
 * one bulk transfer -- avoiding n individual scattered DDR stores. */
void candidate_kernel(const int8_t *values, const int32_t *idx, int32_t *out, int n);
#endif
