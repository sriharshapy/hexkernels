#include <stdint.h>
/* WRONG: computes popcount(a[i] & b[i]) — counts common set bits, NOT differing bits.
   Fails for any pair where a[i] & b[i] != a[i] ^ b[i], e.g. a=0x01, b=0x03:
   AND gives 1 (0x01), XOR gives 2 (0x02). */
void candidate_kernel(const int32_t *a, const int32_t *b, int32_t *out, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)a[i] & (uint32_t)b[i];  /* WRONG: AND instead of XOR */
        v = v - ((v >> 1) & 0x55555555u);
        v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
        v = (v + (v >> 4)) & 0x0F0F0F0Fu;
        out[i] = (int32_t)((v * 0x01010101u) >> 24);
    }
}
