#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 1D FIR SAME-padded (correlation, centered zero-padding):
   Padding: pad_left = (ntaps-1)/2, pad_right = ntaps/2  (for odd ntaps, centered).
   out[i] = sum_{j=0}^{ntaps-1} padded_x[i+j] * taps[j],  i in [0,n).
   padded_x[k] = x[k - pad_left] if 0 <= k-pad_left < n, else 0.
   Equivalently:  out[i] = sum_{j=0}^{ntaps-1}
                              (i - pad_left + j >= 0 && i - pad_left + j < n
                               ? x[i - pad_left + j] : 0) * taps[j].
   x has n samples. out has n int32 results. taps NOT reversed.
   ntaps must be odd so centering is unambiguous (ntaps=7 in harness). */
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int n, int ntaps);
#endif
