/* Near-miss: uses only tableB for ALL inputs (ignores tableA for negatives).
   Wrong for any in[i] < 0 because tableB is indexed 0..127 but
   negative inputs need tableA via the (uint8_t)in[i]-128 index. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n,
                      const int8_t *tableA, const int8_t *tableB){
    (void)tableA;
    for (int i = 0; i < n; i++) out[i] = tableB[(uint8_t)in[i] & 0x7F];
}
