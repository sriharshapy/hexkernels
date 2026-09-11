/* i8_lut_clamp_idx sol_03: HVX clamp of uint8 values then scalar gather
   Use Q6_Vub_vmax_VubVub / Q6_Vub_vmin_VubVub to clamp 128 bytes per vector. */
#include "kernel_api.h"
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, int8_t *out, int n,
                      const int8_t *lut, uint8_t lo, uint8_t hi) {
    const int vlen = 128;
    HVX_Vector vlo = Q6_Vb_vsplat_R((int)(lo | (lo << 8) | (lo << 16) | (lo << 24)));
    HVX_Vector vhi = Q6_Vb_vsplat_R((int)(hi | (hi << 8) | (hi << 16) | (hi << 24)));
    int i = 0;
    for (; i + vlen <= n; i += vlen) {
        HVX_Vector vin = *(const HVX_Vector *)(in + i);
        HVX_Vector vclamped = Q6_Vub_vmax_VubVub(vin, vlo);
        vclamped = Q6_Vub_vmin_VubVub(vclamped, vhi);
        const uint8_t *cp = (const uint8_t *)&vclamped;
        for (int j = 0; j < vlen; j++)
            out[i + j] = lut[cp[j]];
    }
    for (; i < n; i++) {
        uint8_t idx = (uint8_t)in[i];
        if (idx < lo) idx = lo; else if (idx > hi) idx = hi;
        out[i] = lut[idx];
    }
}