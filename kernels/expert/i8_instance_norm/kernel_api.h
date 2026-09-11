#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Instance Normalization: normalize each channel's spatial map independently.
 * int8 in/out, NO floating point.
 *
 * Layout: x[c * HW + h * W + w], row-major, C-last = channel-first here.
 * HW = H * W = spatial elements per channel.
 *
 * Pinned integer formula per channel c (SHIFT=8):
 *   count = HW
 *   1. mu    = sum_s x[c*HW + s] / count           (truncation toward zero)
 *   2. var   = sum_s (x[c*HW+s] - mu)^2 / count    (truncation, >= 0)
 *   3. v_idx = clamp(var, 0, 255)
 *   4. inv   = inv_lut[c * 256 + v_idx]             (uint8, runtime; per-channel LUT)
 *   5. For each spatial position s in [0, HW):
 *      a. d      = (int32)x[c*HW + s] - mu
 *      b. scaled = (d * (int32)gamma[c] + 64) >> 7    (round-half-up)
 *      c. normed = (scaled * (int32)inv + 128) >> 8   (round-half-up, SHIFT=8)
 *      d. out[c*HW + s] = clamp(normed + (int32)beta[c], -128, 127)
 *
 * Parameters:
 *   H=16, W=16, C=16, HW=256, n=C*HW=4096.
 *   gamma   = int8[C] per-channel scale (runtime)
 *   beta    = int8[C] per-channel bias  (runtime)
 *   inv_lut = uint8[C * 256] per-channel LUT (runtime); channel c uses inv_lut + c*256
 */
void candidate_kernel(const int8_t *x, int8_t *out,
                      int H, int W, int C,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut);
#endif /* KERNEL_API_H */
