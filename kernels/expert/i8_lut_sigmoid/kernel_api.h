#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 256-entry int8->int8 lookup: out[i] = lut[(uint8_t)in[i]].
   lut has 256 entries indexed by the input byte reinterpreted as unsigned (0..255). */
void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut);
#endif
