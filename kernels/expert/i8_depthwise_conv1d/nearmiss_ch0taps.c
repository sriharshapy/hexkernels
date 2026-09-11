/* Near-miss: uses channel-0 taps for ALL channels instead of per-channel taps.
   Correct only for ch=0; wrong for ch=1,2,3. Compiles cleanly. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int L, int C, int ntaps) {
    int xstride = L + ntaps - 1;
    /* BUG: always reads taps for channel 0 */
    const int8_t *tap0 = taps;   /* ch0 taps only */
    for (int ch = 0; ch < C; ch++) {
        const int8_t *xch = x   + ch * xstride;
        int32_t      *och = out + ch * L;
        for (int i = 0; i < L; i++) {
            int32_t acc = 0;
            for (int j = 0; j < ntaps; j++)
                acc += (int32_t)xch[i + j] * (int32_t)tap0[j];
            och[i] = acc;
        }
    }
}
