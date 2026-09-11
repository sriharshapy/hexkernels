/* Scalar baseline: int8 elementwise multiply, low-byte wrap.
 *
 * NOTE: the harness's HVX reference (vmul_wrap in harness.c) widens each
 * 128-byte block via Q6_Wh_vsxt_Vb, which DEINTERLEAVES into even-indexed
 * and odd-indexed lanes (not a first-half/second-half split), then packs
 * results back with Q6_Vb_vpacke_VhVh as [even-lane results][odd-lane
 * results]. The actual golden ref[] the candidate is checked against is
 * therefore permuted within each 128-element block:
 *   ref[base + j]      = wrap(a[base + 2*j],   b[base + 2*j])    for j in [0,64)
 *   ref[base + 64 + j] = wrap(a[base + 2*j+1], b[base + 2*j+1])  for j in [0,64)
 * This is transcribed literally (bit-exact vs the harness), not the naive
 * out[i]=wrap(a[i],b[i]) the kernel_api.h prose alone would suggest. */
#include <stdint.h>

static inline int8_t wrap_mul(int8_t x, int8_t y) {
    return (int8_t)((int)x * (int)y);
}

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    const int vlen = 128;
    int i = 0;
    for (; i + vlen <= n; i += vlen) {
        for (int j = 0; j < 64; j++) {
            int idx_even = i + 2*j;
            int idx_odd  = i + 2*j + 1;
            out[i + j]      = wrap_mul(a[idx_even], b[idx_even]);
            out[i + 64 + j] = wrap_mul(a[idx_odd],  b[idx_odd]);
        }
    }
    for (; i < n; i++) out[i] = wrap_mul(a[i], b[i]);
}
