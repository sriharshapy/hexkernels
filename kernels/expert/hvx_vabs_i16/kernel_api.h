#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise int16 absolute value, n=513 (tail path: 513 = 8*64 + 1).
 * Semantics: out[i] = (int16_t)abs((int)a[i])  for i in [0, n), EXCEPT the
 * two's-complement corner case INT16_MIN (-32768): its magnitude (32768)
 * does not fit in int16, so the hardware WRAPS: abs(-32768) = -32768
 * (matches Q6_Vh_vabs_Vh -- the non-saturating abs; NOT Q6_Vh_vabs_Vh_sat,
 * which would clamp to 32767).
 */
void candidate_kernel(const int16_t *a, int16_t *out, int n);
#endif
