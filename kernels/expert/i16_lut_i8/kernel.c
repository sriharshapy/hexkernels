/* i16_lut_i8 sol_04: HVX vector body: clamp int16 lanes then scalar gather
   Use Q6_Vh_vmax_VhVh / Q6_Vh_vmin_VhVh to clamp 64 int16 lanes per vector,
   then gather from lut in scalar (vlut32 is byte-indexed, not i16-indexed). */
#include "kernel_api.h"
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *in, int8_t *out, int n,
                      const int8_t *lut, int lutsize) {
    const int vlen = 128; /* bytes per HVX vector */
    const int elems_per_vec = vlen / 2; /* 64 int16 per vector */
    int hi_val = lutsize - 1;

    /* Broadcast clamp limits into int16 vectors */
    HVX_Vector vzero = Q6_V_vzero();
    /* Pack hi_val into every int16 lane via splat trick */
    HVX_Vector vhi = Q6_Vh_vsplat_R((int16_t)hi_val);

    int i = 0;
    for (; i + elems_per_vec <= n; i += elems_per_vec) {
        HVX_Vector vin = *(const HVX_Vector *)((const int8_t *)in + i * 2);
        HVX_Vector vclamped = Q6_Vh_vmax_VhVh(vin, vzero);
        vclamped = Q6_Vh_vmin_VhVh(vclamped, vhi);
        /* Scalar gather from clamped values */
        const int16_t *clamp_ptr = (const int16_t *)&vclamped;
        for (int j = 0; j < elems_per_vec; j++)
            out[i + j] = lut[clamp_ptr[j]];
    }
    /* Tail */
    for (; i < n; i++) {
        int idx = (int)in[i];
        if (idx < 0) idx = 0; else if (idx > hi_val) idx = hi_val;
        out[i] = lut[idx];
    }
}