#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Quantized SiLU (x*sigmoid(x)) activation via a 256-entry DIRECT-indexed
 * lookup table. n=1000 (NOT a multiple of 128 -- tail path). In-cache
 * working set.
 *
 * Semantics: idx = (uint8_t)x[i]  (reinterpret the signed int8 bit pattern
 * directly as an unsigned index in [0,255] -- no scale/shift arithmetic);
 * out[i] = lut[idx].
 *
 * `lut` is a runtime 256-entry table built by the harness to genuinely
 * represent quantized SiLU: pick a fixed dequant scale S (e.g. 0.05f, so
 * int8 [-128,127] covers input domain [-6.4,6.35]); for k in [0,255],
 * dequantize k as a signed int8 value v=(k>=128?k-256:k), compute
 * silu(v*S) = (v*S) / (1 + exp(-(v*S))), requantize by the same scale S,
 * round-to-nearest, clamp to int8. The table is a runtime input -- do NOT
 * hardcode any activation math inside candidate_kernel; it is a pure gather.
 */
void candidate_kernel(const int8_t *x, int8_t *out, int n, const int8_t *lut);
#endif /* KERNEL_API_H */
