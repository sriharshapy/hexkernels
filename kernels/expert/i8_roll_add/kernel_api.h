#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Circular-roll then add: out[i] = (int8_t)(a[(i + k) % n] + b[i])
 * The addition is two's-complement wraparound (not saturating).
 * k is a runtime parameter; do NOT hardcode it. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n, int k);
#endif
