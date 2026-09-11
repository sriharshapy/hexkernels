/*
 * Near-miss: uses signed integer index (int8_t cast to int) + 128 offset
 * instead of the correct (uint8_t) reinterpretation.
 * For in[i] >= 0 this gives the same index; for in[i] < 0 it also coincidentally
 * gives index = in[i]+128, which equals (uint8_t)in[i].
 * HOWEVER: the intent here is to show the wrong approach that some implementations
 * use: they explicitly add 128 rather than relying on unsigned reinterpretation.
 * If the compiler promotes (int)in[i] as signed, the index arithmetic may differ
 * when the addition overflows or when in[i]==-128 (int(-128)+128==0, correct by
 * accident for this case -- but semantically wrong approach).
 *
 * A more dangerous variant: index with in[i] directly (no offset/cast) -- correct
 * for 0..127 but accesses lut[-128..-1] (undefined behaviour / out of bounds) for
 * negative inputs.
 */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut) {
    /* Wrong: treats in[i] as signed and indexes raw -- UB for negative values
     * and produces wrong output (negative index into lut). */
    for (int i = 0; i < n; i++)
        out[i] = lut[(int)in[i] < 0 ? 0 : (int)in[i]];  /* clips negatives to 0 */
}
