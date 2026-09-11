Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, uint8_t *out, int R, int C, const uint8_t *exp_lut);

Compute row-wise integer softmax over a 2D int8 matrix [R x C], producing uint8 output.
Softmax is applied independently over each ROW of length C.
Pinned multi-step formula per row r (NO floating point):
  1. m_r    = max(x[r*C + j],  j=0..C-1)
  2. idx_j  = clamp((int)(x[r*C+j] - m_r), -255, 0) + 255    (int in [0,255])
  3. e_j    = exp_lut[idx_j]                                   (uint8, runtime input)
  4. S_r    = sum of e_j over j                                (int32)
  5. out[r*C+j] = (uint8)((e_j * 255 + S_r/2) / S_r)          (round-half-down: S_r/2 uses integer /2)

R=8, C=137. C is NOT a multiple of 128 — handle the tail. exp_lut has 256 uint8 entries.
Must read exp_lut at runtime (anti-hardcode). Must iterate over ROWS (not columns).
Prefer HVX vector intrinsics; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT use floating point. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
