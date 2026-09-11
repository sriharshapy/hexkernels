#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 elementwise SIGNED min: out[i] = a[i] < b[i] ? a[i] : b[i] (signed comparison). */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
