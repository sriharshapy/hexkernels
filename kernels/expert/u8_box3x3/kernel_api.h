#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 3x3 box blur, clamp-to-edge borders, out[y*w+x] = (sum of 9 neighbours)/9, same size. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);
#endif
