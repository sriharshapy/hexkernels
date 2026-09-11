Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int16_t *x, int16_t *out, int R, int C, const uint16_t *exp_lut);

Compute row-wise integer softmax over a 2D int16 matrix [R x C], producing int16 output.
Softmax is applied independently over each ROW of length C. NO floating point.

Pinned multi-step formula per row r:
  1. m_r    = max(x[r*C + j],  j=0..C-1)                          (int16)
  2. idx_j  = clamp((int32_t)(x[r*C+j] - m_r), -255, 0) + 255      (int in [0,255])
  3. e_j    = exp_lut[idx_j]                                       (uint16, runtime input)
  4. S_r    = sum of e_j over j                                    (int32)
  5. out[r*C+j] = (int16_t)(((int64_t)e_j*32767 + S_r/2) / S_r)   (round-half-down: S_r/2 uses integer /2)

R=6, C=113. C is NOT a multiple of 128 -- handle the tail. exp_lut has 256 uint16 entries.
Note the output is rescaled to 32767 (not 255) -- a 16-bit fixed-point probability, matching
the wider int16 output container.
Must read exp_lut at runtime (anti-hardcode). Must iterate over ROWS (not columns).
Prefer HVX vector intrinsics; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT use floating point. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
