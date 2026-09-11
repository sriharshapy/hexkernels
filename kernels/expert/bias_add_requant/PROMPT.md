Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *a, const int32_t *bias, int8_t *out, int n,
                          int32_t mult, int shift, int8_t zp);
For each i: compute int64 sum = a[i]+bias[i] (bias is an int32 vector, same length n),
then compute (int64_t)sum * mult, arithmetic-right-shift by `shift` rounding half
AWAY FROM ZERO, add zp, then saturate to int8 [-128,127]. Apply mult in 64-bit BEFORE
the shift. n is not a multiple of 128 — handle the tail. Use HVX intrinsics; include
<hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
