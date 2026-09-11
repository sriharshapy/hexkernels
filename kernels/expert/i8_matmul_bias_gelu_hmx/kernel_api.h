#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 64x64 matmul on the HMX matrix engine with per-column int32 bias add +
 * GELU-via-LUT FUSED into the HVX epilogue:
 *   acc[i][j] = sum_k A[i*n+k] * B[k*n+j]          (A uint8 0..7, B int8 -3..3, n=64)
 *   r[i][j]   = sign_extend_12bit((acc*17+8)>>4)   (HMX 0x40-config requant field)
 *   biased    = r[i][j] + bias[j]                    (int32, per-output-column bias)
 *   pre_lut   = saturate_i8(biased)                  (clamp to [-128,127])
 *   out[i][j] = gelu_lut[(uint8_t)(pre_lut + 128)]    (256-entry int8 LUT, index=val+128)
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output crouton tiles. bias is [n] int32,
 * gelu_lut is [256] int8 (a RUNTIME parameter -- do not hardcode it).
 * Output is bit-exact int8 (the LUT is the exact contract, no tolerance needed).
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias,
                       const int8_t *gelu_lut, int8_t *out, int n);
#endif
