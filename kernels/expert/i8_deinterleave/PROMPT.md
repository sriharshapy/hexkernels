Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *in, int8_t *a, int8_t *b, int n);

Deinterleave in[2*n] into two arrays of length n:
 a[i] = in[2*i]
 b[i] = in[2*i+1]

n=500 (not a multiple of 128 — handle the tail). Pure data movement, no arithmetic.
The HVX intrinsic `Q6_W_vshuff_VVR` with shift=1 deinterleaves two 128-byte vectors.
Respond with a single complete C code block and CLOSE the fence with ```.
