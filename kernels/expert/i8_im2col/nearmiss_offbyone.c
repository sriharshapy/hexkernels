/* Near-miss: off-by-one in kernel window — uses (oh + kh) and (ow + kw) instead of
   (oh*S + kh) and (ow*S + kw). Equivalent for S=1 at first glance, but gets the
   base wrong when oh or ow starts at 0 vs 1 for boundary positions. Actually for S=1
   this is the same — so the real bug here is kh/kw channel ordering: uses wrong row
   formula c + kh*C*K + kw*C instead of c*K*K + kh*K + kw, swapping channel order. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out,
                      int C, int H, int W,
                      int K, int S,
                      int OH, int OW) {
    (void)H;
    /* BUG: row formula swaps channel-major with kernel-major order */
    for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++)
            for (int c = 0; c < C; c++) {
                int row = kh*K*C + kw*C + c;  /* BUG: should be c*K*K + kh*K + kw */
                for (int oh = 0; oh < OH; oh++)
                    for (int ow = 0; ow < OW; ow++) {
                        int col = oh*OW + ow;
                        out[row * OH*OW + col] = in[c*H*W + (oh*S + kh)*W + (ow*S + kw)];
                    }
            }
}
