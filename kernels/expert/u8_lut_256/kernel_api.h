#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 256-entry uint8->uint8 lookup: out[i] = lut[in[i]].
   in[] is UNSIGNED (0..255); index needs no reinterpretation.
   lut has 256 entries supplied as a runtime pointer. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int n, const uint8_t *lut);
#endif
