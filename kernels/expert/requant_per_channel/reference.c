#include <stdint.h>
void candidate_kernel(const int32_t *a, int8_t *out, int R, int C,
                      const int32_t *mult, const int *shift, int8_t zp) {
    for (int r = 0; r < R; r++) {
        int32_t m = mult[r];
        int      s = shift[r];
        int64_t  half = s > 0 ? ((int64_t)1 << (s - 1)) : 0;
        for (int c = 0; c < C; c++) {
            int64_t v = (int64_t)a[r * C + c] * (int64_t)m;
            int64_t r2 = (v >= 0) ? ((v + half) >> s) : -(((-v) + half) >> s);
            r2 += zp;
            if (r2 > 127) r2 = 127;
            if (r2 < -128) r2 = -128;
            out[r * C + c] = (int8_t)r2;
        }
    }
}
