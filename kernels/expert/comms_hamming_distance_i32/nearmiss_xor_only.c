#include <stdint.h>
/* WRONG: returns a[i] ^ b[i] directly — the XOR bitmask, NOT the count of differing bits.
   Fails for any pair where the XOR value != the popcount of that XOR, e.g.
   a=0x03, b=0x00: XOR=3 but Hamming distance=2. */
void candidate_kernel(const int32_t *a, const int32_t *b, int32_t *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] ^ b[i];  /* WRONG: XOR word, not its popcount */
    }
}
