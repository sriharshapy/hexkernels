Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n,
                          int32_t mult, int shift, int8_t zp);
Fused elementwise: for each i in [0, n) compute:
  Step 1 -- add:     sum = a[i] + b[i]              (int32/int64 to avoid overflow)
  Step 2 -- relu:    sum = max(sum, 0)
  Step 3 -- requantize to int8:
    v    = (int64_t)sum * mult
    half = shift > 0 ? (1LL << (shift-1)) : 0
    r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)   // round half away from 0
    r   += zp
    out[i] = saturate_to_int8(r)                    // clamp to [-128, 127]
n=1024 (NOT a multiple of 128) -- handle the tail. mult, shift, zp are runtime params (do NOT hardcode).
Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
