#include <stdint.h>
/* Bitwise Hamming distance per word pair: out[i] = popcount( a[i] ^ b[i] ).
   Standard Hamming-weight algorithm, counts all 32 bits. */
void candidate_kernel(const int32_t *a, const int32_t *b, int32_t *out, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)a[i] ^ (uint32_t)b[i];
        v = v - ((v >> 1) & 0x55555555u);
        v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
        v = (v + (v >> 4)) & 0x0F0F0F0Fu;
        out[i] = (int32_t)((v * 0x01010101u) >> 24);
    }
}
