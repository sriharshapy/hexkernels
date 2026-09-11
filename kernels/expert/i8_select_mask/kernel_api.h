#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Conditional select (where): out[i] = mask[i] ? a[i] : b[i].
   mask[i] is nonzero-true (any non-zero value selects a[i]; zero selects b[i]).
   a and b are signed int8; out is signed int8. */
void candidate_kernel(const int8_t *a, const int8_t *b, const uint8_t *mask,
                      int8_t *out, int n);
#endif
