#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * 256-entry int8->int8 lookup table (tanh framing).
 * out[i] = lut[(uint8_t)in[i]]
 * The index is the input byte reinterpreted as UNSIGNED (0..255):
 *   in[i] in [-128,127]  ->  index in [0,255]
 *   e.g. in=-128 -> index 128, in=-1 -> index 255, in=0 -> index 0.
 * lut has exactly 256 entries.  n=1024 (not a multiple of 128; handle the tail).
 */
void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut);
#endif
