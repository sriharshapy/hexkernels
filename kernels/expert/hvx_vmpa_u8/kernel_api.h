#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Dual weighted-sum multiply-add pair, n=780 (tail path: 780 = 6*128 + 12).
 * Semantics: out[i] = (int16_t)((int)a[i]*(int)wa[i] + (int)b[i]*(int)wb[i])
 * for i in [0, n). All of a,b,wa,wb are uint8. Two's-complement WRAP on
 * int16 overflow (the hardware MAC does not saturate). Matches HVX
 * Q6_Wh_vmpa_WubWub applied per-lane across the (a,wa) and (b,wb) operand
 * pairs (NOT a 4-tap group reduction like vdmpy -- this is a per-index
 * two-term weighted sum of two channels).
 */
void candidate_kernel(const uint8_t *a, const uint8_t *b,
                      const uint8_t *wa, const uint8_t *wb,
                      int16_t *out, int n);
#endif
