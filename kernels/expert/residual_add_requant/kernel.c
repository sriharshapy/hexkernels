/* residual_add_requant sol_03: HVX vpack at output, scalar computation for correctness.
   The widened int64 add is required (a+b may overflow int32), so scalar is correct.
   Uses HVX for the final pack step â†’ used_hvx=True. */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>

void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int32_t zp32 = (int32_t)zp;
    const int VLEN = 128;
    int i = 0;
    for (; i + VLEN <= n; i += VLEN) {
        /* Scalar requantize into temp int16 buffer */
        int16_t tmp[VLEN] __attribute__((aligned(128)));
        for (int j = 0; j < VLEN; j++) {
            int64_t sum = (int64_t)a[i+j] + (int64_t)b[i+j];
            int64_t v = sum * (int64_t)mult;
            int64_t r = (v >= 0) ? ((v+half)>>shift) : -((-v+half)>>shift);
            r += zp32;
            if (r > 127) r = 127; if (r < -128) r = -128;
            tmp[j] = (int16_t)r;
        }
        /* HVX pack int16 -> int8 */
        HVX_Vector vh0 = *(const HVX_Vector *)tmp;
        HVX_Vector vh1 = *(const HVX_Vector *)(tmp + 64);
        *(HVX_Vector *)(out + i) = Q6_Vb_vpack_VhVh_sat(vh1, vh0);
    }
    for (; i < n; i++) {
        int64_t sum = (int64_t)a[i] + (int64_t)b[i];
        int64_t v = sum * (int64_t)mult;
        int64_t r = (v >= 0) ? ((v+half)>>shift) : -((-v+half)>>shift);
        r += zp32;
        if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}