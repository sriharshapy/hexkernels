#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Per-channel requantize int32 -> int8:
 * Layout: `a` is a flat array of R rows x C columns (row-major), total R*C elements.
 * Each row r uses mult[r] and shift[r] (per-channel scale), shared zero-point `zp`.
 *   out[r*C+c] = round_half_away_from_zero( (int64_t)a[r*C+c]*mult[r] >> shift[r] ) + zp,
 *                saturated to int8 [-128,127].
 * mult[] and shift[] arrays each have R entries. shift[r] >= 0. */
void candidate_kernel(const int32_t *a, int8_t *out, int R, int C,
                      const int32_t *mult, const int *shift, int8_t zp);
#endif
