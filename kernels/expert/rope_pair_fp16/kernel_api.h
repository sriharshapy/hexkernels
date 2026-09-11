#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/*
 * Standalone rotary position embedding (RoPE) on interleaved element pairs,
 * fp16 (float-domain, no fixed-point).
 *
 * x:   [2*n_pairs] fp16, interleaved (x[2p], x[2p+1]) = one rotation pair.
 * cos: [n_pairs] fp16 -- per-pair cosine.
 * sin: [n_pairs] fp16 -- per-pair sine.
 * out: [2*n_pairs] fp16, interleaved, same layout as x.
 *
 * For each pair index p in [0, n_pairs), let d=2*p, d2=2*p+1:
 *   out[d]  = (float)x[d]*(float)cos[p] - (float)x[d2]*(float)sin[p]
 *   out[d2] = (float)x[d]*(float)sin[p] + (float)x[d2]*(float)cos[p]
 *
 * n_pairs=100 (x has 200 fp16 elements -- not a multiple of 64 fp16 lanes
 * per HVX vector, a real tail path). Output compared with fp16 tolerance
 * (HVX float arithmetic is non-IEEE qfloat).
 */
void candidate_kernel(const hvx_hf *x, const hvx_hf *cos, const hvx_hf *sin,
                      hvx_hf *out, int n_pairs);
#endif /* KERNEL_API_H */
