Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *a, int8_t *out, int R, int C,
                          const int32_t *mult, const int *shift, int8_t zp);
Layout: `a` and `out` are R rows x C columns, row-major (R=40, C=25 in the harness).
For each row r and column c:
  - compute (int64_t)a[r*C+c] * mult[r]  (per-row multiplier, applied in 64-bit)
  - arithmetic-right-shift by shift[r] rounding half AWAY FROM ZERO
  - add shared zero-point `zp`
  - saturate to int8 [-128,127]
  - store to out[r*C+c]
Each row has its own mult[r] and shift[r] (per-channel scale). shift[r] >= 0.
R*C is not a multiple of 128 — handle the tail. Use HVX intrinsics; include
<hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
