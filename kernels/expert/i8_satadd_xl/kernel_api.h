#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 elementwise SIGNED SATURATING add: out[i] = clamp(a[i]+b[i], -128, 127).
 * This is NOT two's-complement wraparound (matches Q6_Vb_vadd_VbVb_sat). */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
