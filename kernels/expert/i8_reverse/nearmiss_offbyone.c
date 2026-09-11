/* Near-miss: off-by-one — uses in[n-i] instead of in[n-1-i].
   For i=0 this reads in[n] which is out-of-bounds; for i>0 it is shifted by 1.
   The output is shifted by one position, failing all but possibly the last element. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = in[n - i];  /* BUG: should be in[n-1-i]; reads OOB for i=0 */
}
