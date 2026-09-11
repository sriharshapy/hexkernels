Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out, int n,
                          int32_t mult, int shift, int8_t zp);

Implement fused gather + requantize for each output position i in [0, n):
    raw    = (int32_t)table[ idx[i] ]       -- gather from table
    v      = raw * mult                      -- scale (use int64 intermediate)
    half   = shift > 0 ? (1LL << (shift-1)) : 0
    r      = (v >= 0) ? (v + half) >> shift
                      : -(((-v) + half) >> shift)   -- round-half-away-from-zero
    r     += zp                              -- zero-point offset
    out[i] = saturate_to_int8(r)            -- clamp to [-128, 127]

table: lookup table, [N_TABLE=256] int8, 128-byte aligned.
idx:   gather indices,[n=512]      int32, 128-byte aligned. Values in [0, 256).
out:   output,        [n=512]      int8, 128-byte aligned.
mult, shift, zp: RUNTIME parameters -- do NOT hardcode them.
n=512 is NOT a multiple of 128 -- handle the tail correctly.

The index array and parameters are all runtime inputs.
The harness sweeps multiple (mult, shift, zp) sets, so hardcoding any of them
causes test failures.

Do NOT write main(). Do NOT hardcode index values or quantization parameters.
Respond with a single complete C code block and CLOSE the fence with ```.
