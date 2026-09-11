#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Standalone 2D bias-add epilogue (no requant): add a per-COLUMN bias vector to
 * every row of an [R x C] int32 tile.
 *
 *   out[r*C+c] = acc[r*C+c] + bias[c]     for all r in [0,R), c in [0,C)
 *
 * `bias` is int32[C], BROADCAST across all R rows (genuinely 2D layout, not a
 * flat 1D elementwise add). Output stays int32 -- no saturation/requant here
 * (that's epilogue_requant_i32_i8's concern); plain int32 add, overflow is out
 * of scope (test inputs are chosen so acc[r*C+c]+bias[c] does not overflow
 * int32). */
void candidate_kernel(const int32_t *acc, const int32_t *bias, int32_t *out, int R, int C);
#endif
