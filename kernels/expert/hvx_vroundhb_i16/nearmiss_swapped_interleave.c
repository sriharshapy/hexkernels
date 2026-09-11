/* NEAR-MISS: swaps the a/b interleave order, passing (b,a) into
 * Q6_Vb_vround_VhVh_sat instead of (a,b). Compiles, and happens to match
 * only when a[i]==b[i] pairwise, but fails bit-exact on generic inputs
 * where the even/odd rounded values differ. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int n) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + 2*i) = Q6_Vb_vround_VhVh_sat(vb, va);  /* WRONG: swapped */
    }
    for (; i < n; i++) {
        int rb = ((int)b[i] + 128) >> 8;
        int ra = ((int)a[i] + 128) >> 8;
        if (ra > 127) ra = 127; if (ra < -128) ra = -128;
        if (rb > 127) rb = 127; if (rb < -128) rb = -128;
        out[2*i]     = (int8_t)ra;  /* WRONG: swapped */
        out[2*i + 1] = (int8_t)rb;  /* WRONG: swapped */
    }
}
