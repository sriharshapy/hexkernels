#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* L3 fp16 FLASH-ATTENTION TILE -- one tiled flash-attention step for a Q tile:
 * stream K/V in tiles of FLASH_TILE_K rows, maintaining the online-softmax
 * recurrence (running max m, running sum l, running weighted-V accumulator
 * acc), and rescaling the running accumulator/sum whenever the running max
 * increases. Mathematically produces EXACTLY the standard (non-flash, full
 * softmax) attention output for that Q tile -- the flash recurrence is an
 * algorithmic reformulation, not a different function.
 *
 *   scale = FLASH_SCALE
 *   for each query row i in [0,SQ):
 *     m = -INFINITY; l = 0; acc[0..DH-1] = 0
 *     for each K/V tile t of FLASH_TILE_K rows (SK / FLASH_TILE_K tiles):
 *       for k in tile:  s[k] = scale * dot(Q[i,:], K[t*FLASH_TILE_K+k,:])
 *       tile_max = max_k s[k]
 *       new_m    = max(m, tile_max)
 *       corr     = exp(m - new_m)              (rescale factor for OLD state; 0 on first tile since m=-INF)
 *       l   = l * corr
 *       acc[:] = acc[:] * corr
 *       for k in tile:
 *         p = exp(s[k] - new_m)
 *         l += p
 *         acc[:] += p * V[t*FLASH_TILE_K+k, :]
 *       m = new_m
 *     O[i,:] = acc[:] / l
 *
 * This equals softmax(Q.K^T * scale) . V computed the standard (non-tiled) way;
 * the reference (harness) computes it the standard way for cross-check. HVX
 * float goes through the non-IEEE qf16 path and float ops reorder, so the
 * compare uses hvx_close_f16bits (tolerance), not bit-exactness.
 *
 * SQ=16 (query rows), SK=64 (key/value rows), DH=64 (head dim -- exactly one
 * 64-lane fp16 HVX vector, so every Q/K/V row is one full vector, no padding).
 * FLASH_TILE_K=16 -> SK/FLASH_TILE_K = 4 tiles.
 */
#define FLASH_TILE_K 16
#define FLASH_SCALE  0.125f

void candidate_kernel(const hvx_hf *Q, const hvx_hf *K, const hvx_hf *V,
                      hvx_hf *O, int SQ, int SK, int DH);
#endif
