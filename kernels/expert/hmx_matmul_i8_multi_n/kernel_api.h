#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 32x64 matmul (multiple N-tiles) on the HMX matrix engine.
 *   acc[i][j] = sum_k A[i*K+k] * B[k*ncol+j]         (A uint8 0..7, B int8 -3..3)
 *   out[i][j] = sign_extend_12bit((acc*17+8)>>4)     (M=K=32, N=ncol=64, int32)
 * The output width N=64 spans TWO 32-wide crouton output tiles; run the 32x32
 * HMX matmul once per N-tile (weight columns tj*32 .. tj*32+31). The harness
 * enables the HMX context; use VTCM scratch at HVX_VTCM_BASE. int32, bit-exact. */
void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out,
                       int m, int ncol, int k);
#endif
