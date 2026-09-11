#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-block concatenation of two int16 half-blocks via vcombine, g=3
 * (3 full 64-element blocks per input; block-granular op, no scalar
 * tail). Each of a, b has n = 64*g int16 elements.
 * Semantics: for each block k (64 elements at offset k*64), and for
 * i in [0, 64):
 *   out[k*128 + i]      = a[k*64 + i]
 *   out[k*128 + 64 + i] = b[k*64 + i]
 * i.e. out is the natural concatenation [a_block, b_block] per block
 * (out has 2*64*g = 128*g int16 elements total).
 * Matches Q6_W_vcombine_VV(Vu, Vv) which forms a 256-byte VectorPair
 * whose LOW half (Q6_V_lo_W) is Vv and whose HIGH half (Q6_V_hi_W) is
 * Vu -- so to get out=[a,b] you must call vcombine(b, a) and store
 * lo=a first, hi=b second (pinned via sim experiment: lo of vcombine
 * always equals the SECOND intrinsic argument).
 */
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int g);
#endif
