#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * 256-entry byte lookup table, n=900 (tail path: 900 = 7*128 + 4).
 * Semantics: idx = (uint8_t)a[i]  (reinterpret the signed byte's bit
 * pattern as an unsigned index in [0,255]); out[i] = lut[idx].
 * lut has exactly 256 entries, supplied as a runtime pointer (do NOT
 * hardcode values -- the table is a task input, matching HVX vlut32
 * which is a general byte->byte gather, not a fixed nonlinearity).
 */
void candidate_kernel(const int8_t *a, int8_t *out, int n, const int8_t *lut);
#endif
