Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                          const int16_t *gamma, const int16_t *beta,
                          const uint16_t *inv_lut);

Compute batched integer LayerNorm over R independent rows of length C
(x/out are [R x C] row-major). gamma/beta are length-R and PER-ROW: gamma[r]
and beta[r] are scalar values broadcast uniformly across all C columns of
row r (NOT per-column). NO floating point anywhere. Pinned formula per row r:

  1. mu    = sum_c(x[r][c]) / C                        (int64 sum -> int32 mu,
                                                          truncation toward zero)
  2. var   = sum_c((x[r][c]-mu)^2) / C                   (int64 sum/accum -> int64 var,
                                                          truncation, >=0)
  3. vidx  = clamp(var >> 5, 0, 255)                      (LUT index)
  4. inv   = inv_lut[vidx]                                (uint16, runtime, must be read
                                                          at runtime -- anti-hardcode)
  5. Per column c:
     a. d       = (int32)(x[r][c] - mu)
     b. normed  = (d * (int32)inv + 512) >> 10             (round-half-up, inv-scale
                                                          applied FIRST)
     c. scaled  = (normed * (int32)gamma[r] + 32) >> 6      (round-half-up, gamma
                                                          applied SECOND, per-row scalar)
     d. out[r][c] = clamp(scaled + (int32)beta[r], -32768, 32767)   (int16 saturate)

R=6, C=90 (C is NOT a multiple of 64 int16-lanes-per-HVX-vector -- handle the tail
path). gamma and beta are int16 length-R runtime arrays (one scalar per row,
broadcast over columns -- do NOT index them by column). inv_lut has 256 uint16
entries (runtime, must be read, not hardcoded).

Prefer HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT use floating point. Implement ONLY this function (Hexagon HVX C).
Do NOT write main(). Respond with a single complete C code block.
