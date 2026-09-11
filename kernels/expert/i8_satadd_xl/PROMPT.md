Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
Compute out[i] = clamp(a[i] + b[i], -128, 127) for i in [0, n) using SIGNED SATURATING
int8 addition (matching `Q6_Vb_vadd_VbVb_sat`; this is NOT two's-complement wraparound —
e.g. 127+1 must saturate to 127, not wrap to -128). n=500000 (working set ~1.5 MB,
DDR-bound); n is not a multiple of 128 — handle the tail.
Respond with a single complete C code block and CLOSE the fence with ```.
