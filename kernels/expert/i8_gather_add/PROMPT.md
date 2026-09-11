Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in_a, const int8_t *in_b,
                          const int32_t *idx, int8_t *out, int n);

Implement fused gather-then-add for each output position i in [0, n):
    out[i] = (int8_t)( (int16_t)in_a[idx[i]] + (int16_t)in_b[i] )

Arithmetic is two's-complement int8 WRAPAROUND (no saturation, no clamping).

in_a: source array to gather from, [N_SRC=256] int8, 128-byte aligned.
in_b: bias array added element-wise, [n=512]    int8, 128-byte aligned.
idx:  gather indices,                [n=512]     int32, 128-byte aligned. Values in [0, 256).
out:  output array,                  [n=512]     int8, 128-byte aligned.
n=512 is NOT a multiple of 128 -- handle the tail correctly.

The index array is a runtime input; do NOT assume contiguous or identity indices.

Hint: gather (random read from in_a) followed by elementwise add with in_b.
Use HVX for the add; gather elements individually then process in vector batches.

Do NOT write main(). Do NOT hardcode index values or ignore in_b.
Respond with a single complete C code block and CLOSE the fence with ```.
