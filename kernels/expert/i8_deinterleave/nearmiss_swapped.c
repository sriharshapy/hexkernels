/* Near-miss: a and b outputs are swapped — a gets odd indices, b gets even.
   Compiles and runs but produces incorrect split. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *a, int8_t *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = in[2*i + 1];  /* BUG: should be in[2*i] */
        b[i] = in[2*i];      /* BUG: should be in[2*i+1] */
    }
}
