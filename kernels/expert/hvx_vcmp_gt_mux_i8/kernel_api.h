#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-lane compare-then-select across THREE inputs, n=1013 (tail path:
 * 1013 = 7*128 + 117).
 * Semantics: out[i] = (a[i] > b[i]) ? a[i] : c[i] for i in [0, n)
 * (signed int8 compare). Note c, not b, is selected on the false branch
 * -- this is a genuine 3-input select, not a 2-input max.
 * Matches Q6_Q_vcmp_gt_VbVb(a, b) to build the predicate, then
 * Q6_V_vmux_QVV(Qt, a, c) to select.
 */
void candidate_kernel(const int8_t *a, const int8_t *b, const int8_t *c, int8_t *out, int n);
#endif
