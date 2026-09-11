Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
 int T, int E);

Implement embedding gather: copy embedding rows by index.
For each token i in [0, T):
 out[i*E + e] = table[idx[i]*E + e] for e in [0, E)

table: [VOCAB_SIZE x E] int8, row-major, 128-byte aligned. VOCAB_SIZE=64, E=128.
idx: [T] int32, valid indices in [0, VOCAB_SIZE).
out: [T x E] int8, row-major, 128-byte aligned.
T=33 (token count, NOT a multiple of 128 — handle the tail).
E=128 (embedding dim — exactly one HVX vector per row).

Prefer HVX vmem loads/stores for whole-row copies (vmemu/vmemw).
Respond with a single complete C code block and CLOSE the fence with ```.
