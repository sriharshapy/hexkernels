Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                          const int16_t *gamma, const int16_t *beta,
                          const uint16_t *inv_lut);

Compute batched integer LayerNorm over R independent rows of length C
(x/out are [R x C] row-major; gamma/beta are length-C and SHARED/broadcast
across all R rows). NO floating point anywhere. Pinned formula per row r:

  1. mu    = sum_c(x[r][c]) / C                        (int64 sum -> int32 mu,
                                                          truncation toward zero)
  2. var   = sum_c((x[r][c]-mu)^2) / C                   (int64 sum/accum -> int64 var,
                                                          truncation, >=0)
  3. vidx  = clamp(var >> 5, 0, 255)                      (LUT index)
  4. inv   = inv_lut[vidx]                                (uint16, runtime, must be read
                                                          at runtime -- anti-hardcode)
  5. Per column c:
     a. d       = (int32)(x[r][c] - mu)
     b. scaled  = (d * (int32)gamma[c] + 32) >> 6          (round-half-up)
     c. normed  = (scaled * (int32)inv + 512) >> 10         (round-half-up)
     d. out[r][c] = clamp(normed + (int32)beta[c], -32768, 32767)   (int16 saturate)

R=6, C=100 (C is NOT a multiple of 64 int16-lanes-per-HVX-vector -- handle the tail
path). gamma and beta are int16 length-C runtime arrays (shared across rows).
inv_lut has 256 uint16 entries (runtime, must be read, not hardcoded). x/out are
int16 (wider dynamic range than the int8 layernorm original).

Prefer HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT use floating point. Implement ONLY this function (Hexagon HVX C). Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main(). Respond with a single complete C code block.
