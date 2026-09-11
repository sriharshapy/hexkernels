#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Bias-add then requantize int32 -> int8:
 *   r = round_half_away_from_zero( (int64_t)(a[i] + bias[i]) * mult >> shift ) + zp,
 *   saturated to int8 [-128,127].
 * `bias` is an int32 array of length n (same as a). The sum is computed in int64.
 * `mult` applied in int64 before arithmetic right shift by `shift` (>=0). */
void candidate_kernel(const int32_t *a, const int32_t *bias, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp);
#endif
