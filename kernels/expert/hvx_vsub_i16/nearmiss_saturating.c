/* NEAR-MISS: uses the SATURATING subtract (Q6_Vh_vsub_VhVh_sat) instead of
 * the wrapping subtract. Compiles and passes for non-overflowing inputs,
 * but fails on the pinned overflow edge cases (e.g. -32768-1 should wrap
 * to 32767, not saturate to -32768). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vh_vsub_VhVh_sat(va, vb);   /* WRONG: saturating */
    }
    for (; i < n; i++) {
        int r = (int)a[i] - (int)b[i];
        if (r > 32767) r = 32767;
        if (r < -32768) r = -32768;
        out[i] = (int16_t)r;
    }
}
