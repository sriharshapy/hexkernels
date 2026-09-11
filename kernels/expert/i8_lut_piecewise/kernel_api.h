#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Dual-table LUT selected by sign of input:
   if in[i] < 0:  idx = (uint8_t)in[i]          (0..127 for -128..-1)
                  out[i] = tableA[idx]
   if in[i] >= 0: idx = in[i]                    (0..127)
                  out[i] = tableB[idx]
   tableA has 128 entries (indexed by (uint8_t)in[i] which is 128..255, but
   stored compactly: tableA[0] = value for in=-128, tableA[127] = value for in=-1;
   i.e., tableA is indexed by (uint8_t)in[i] - 128).
   tableB has 128 entries indexed 0..127 (tableB[0]=value for in=0).
   Both tables supplied as runtime pointers — do NOT hardcode. */
void candidate_kernel(const int8_t *in, int8_t *out, int n,
                      const int8_t *tableA, const int8_t *tableB);
#endif
