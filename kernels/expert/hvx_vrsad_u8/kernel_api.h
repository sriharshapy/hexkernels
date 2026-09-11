#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * 4-wide sum-of-absolute-differences against a FIXED 4-byte pattern
 * {10,20,30,40}, g=98 groups (tail path: 98 groups = 3*32 + 2; input
 * length n = 4*g = 392 uint8 elements).
 * Semantics: for k in [0, g):
 *   out[k] = sum_{j=0..3} |(int)a[4*k+j] - (int)pattern[j]|
 * where pattern = {10, 20, 30, 40} (unsigned byte values).
 * Matches Q6_Wuw_vrsad_WubRubI(Vuu, Rt, Iu1=0) with Vuu built as
 * vcombine(Vu, Vu) (duplicate the single input vector into both halves
 * of the pair) and Rt = the pattern's 4 bytes packed little-endian
 * (byte k of Rt == pattern[k]): Rt = 10 | (20<<8) | (30<<16) | (40<<24)
 * = 0x281E140A. With Iu1=0, Q6_V_lo_W of the result gives, per 32-group
 * (128-byte) input vector, the 32 uint32 SAD-against-pattern values
 * (pinned via sim experiment).
 */
void candidate_kernel(const uint8_t *a, uint32_t *out, int g);
#endif
