#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 32x32 matmul on the HMX matrix engine + requantize-to-int8 with SATURATION:
 *   acc[i][j] = sum_{k} A[i*n+k] * B[k*n+j]              (A uint8 0..15, B int8 -4..4)
 *   r[i][j]   = sign_extend_12bit( (acc*17 + 8) >> 4 )    (0x40-config HMX requant)
 *   out[i][j] = clamp(r[i][j], -128, 127)                  (signed int8, saturating)
 * n == 32. The input range makes |r| exceed 127 often, so the int8 saturation is
 * load-bearing (a plain truncating narrow is wrong). Harness enables the HMX
 * context; stage crouton tiles through VTCM at HVX_VTCM_BASE. Bit-exact int8.
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, int8_t *out, int n);
#endif
