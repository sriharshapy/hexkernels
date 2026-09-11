Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, uint8_t *out, int R, int C,
                          const uint8_t *exp_lut, const uint16_t *recip_lut);

Compute row-wise integer softmax over a 2D int8 matrix [R x C], producing
uint8 output. Softmax is applied independently over each ROW of length C.
This variant AVOIDS integer division entirely (unlike a division-based
row-softmax) by normalizing through a quantized 16-bit reciprocal LUT.

Pinned multi-step formula per row r (NO floating point, NO division):
  1. m      = max(x[r*C+j], j=0..C-1)                              (int8)
  2. idx_j  = clamp((int)x[r*C+j] - m, -255, 0) + 255               (in [0,255])
  3. e_j    = exp_lut[idx_j]                                        (uint8, runtime)
  4. S      = sum_j (int32)e_j, j=0..C-1                            (int32)
  5. sidx   = clamp(S >> 6, 0, 255)                                 (SSHIFT=6)
  6. recip  = recip_lut[sidx]                                       (uint16, runtime)
  7. out[r*C+j] = (uint8_t) clamp(((int32_t)e_j*(int32_t)recip + 32768) >> 16, 0, 255)

R=8, C=100. C is NOT a multiple of 128 -- handle the tail (pad/exclude the
extra lanes from the max and the sum, do not let padding contaminate them).
exp_lut has 256 uint8 entries; recip_lut has 256 uint16 entries. Both must
be read at runtime (anti-hardcode) -- do not assume their contents.
Prefer HVX vector intrinsics; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT use floating point. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
