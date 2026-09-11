#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Fused gather + requantize: out[i] = requant(table[idx[i]])
 *
 * For each output position i in [0, n):
 *   raw   = (int32_t)table[ idx[i] ]  -- gathered int8 widened to int32
 *   v     = raw * mult                -- scale (int64 intermediate)
 *   half  = shift > 0 ? (1LL << (shift-1)) : 0
 *   r     = (v >= 0) ? (v + half) >> shift
 *                    : -(((-v) + half) >> shift)   -- round-half-away-from-zero
 *   r    += zp                        -- zero-point offset
 *   out[i] = saturate_to_int8(r)      -- clamp to [-128, 127]
 *
 * mult, shift, zp are RUNTIME parameters -- do NOT hardcode them.
 * The harness sweeps multiple (mult, shift, zp) sets to catch hardcoding.
 *
 * Constraints:
 *   0 <= idx[i] < N_TABLE  (guaranteed by harness; N_TABLE=256).
 *   Indices are int32.
 *   n = 512 (NOT a multiple of 128 -- handle the tail).
 *
 * table: [N_TABLE=256] int8, 128-byte aligned.
 * idx:   [n=512]       int32, 128-byte aligned.
 * out:   [n=512]       int8, 128-byte aligned.
 */
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp);
#endif /* KERNEL_API_H */
