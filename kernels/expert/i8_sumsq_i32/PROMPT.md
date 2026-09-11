Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *a, int n, int32_t *out);
Compute out[0] = sum over i in [0,n) of a[i]*a[i], accumulated in int32 (signed).
n is not a multiple of 128 — handle the tail. Prefer the HVX vrmpy reduction intrinsics.
Respond with a single complete C code block and CLOSE the fence with ```.
