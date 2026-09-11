/*
 * sol_03: add_relu_requant -- HVX for add+relu+pack, scalar for the 64-bit multiply.
 *
 * Diversity: use HVX to compute add+relu+pack-to-int8 (the SIMD-friendly parts),
 * then do the requantize multiply in scalar. We load int32 with HVX, add, relu,
 * then unpack and scalar-multiply. This uses HVX registers for the add/relu stage.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp)
{
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    HVX_Vector vzero = Q6_V_vzero();

    /* Process 32 elements at a time (1 HVX vector of int32) */
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);

        /* add + relu via HVX */
        HVX_Vector vs = Q6_Vw_vmax_VwVw(Q6_Vw_vadd_VwVw(va, vb), vzero);

        /* unpack to scalar and do 64-bit multiply+requantize */
        int32_t tmp[32] __attribute__((aligned(128)));
        *(HVX_Vector *)tmp = vs;

        for (int j = 0; j < 32; j++) {
            int64_t v = (int64_t)tmp[j] * (int64_t)mult;
            int64_t r = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
            r += zp;
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            out[i + j] = (int8_t)r;
        }
    }

    /* Scalar tail */
    for (; i < n; i++) {
        int64_t sum = (int64_t)a[i] + (int64_t)b[i];
        if (sum < 0) sum = 0;
        int64_t v   = sum * (int64_t)mult;
        int64_t r   = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
        r += zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}