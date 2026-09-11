#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Standalone SwiGLU gating step (elementwise, no matmul -- the gating
 * nonlinearity applied AFTER the gate/up projections, WITHOUT the
 * surrounding matmuls):
 *
 *   idx    = (uint8_t)((int)gate[i] + 128)              (in [0,255])
 *   g      = silu_lut[idx]                                (int8, runtime SiLU-of-gate LUT)
 *   prod   = (int32_t)g * (int32_t)up[i]                   (int32; range +-127*127)
 *   r      = (int64_t)prod * (int64_t)scale_mult
 *   half   = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
 *   q      = (r >= 0) ? ((r + half) >> scale_shift) : -((-r + half) >> scale_shift)
 *   out[i] = clamp(q, -128, 127)
 *
 * gate: [N] int8 (already-projected gate values).
 * up:   [N] int8 (already-projected up values).
 * silu_lut: [256] int8, RUNTIME (represents SiLU(dequantized gate) re-quantized);
 *   index = (uint8_t)(gate[i] + 128). Do NOT hardcode the LUT.
 * scale_mult (int32) / scale_shift (int) are RUNTIME parameters (swept by the
 *   harness) -- do NOT hardcode them. */
void candidate_kernel(const int8_t *gate, const int8_t *up, const int8_t *silu_lut,
                      int8_t *out, int N, int32_t scale_mult, int scale_shift);
#endif /* KERNEL_API_H */
