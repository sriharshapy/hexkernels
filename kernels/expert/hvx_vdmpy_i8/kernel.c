/* expert: same as solutions/s1.c (drill task, accelerable=false; 1.0x is fine). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *a, const int8_t w[4], int16_t *out, int g) {
    uint32_t Rt = ((uint32_t)(uint8_t)w[3] << 24) | ((uint32_t)(uint8_t)w[2] << 16) |
                  ((uint32_t)(uint8_t)w[1] << 8)  |  (uint32_t)(uint8_t)w[0];
    int nvec = g / 32;
    int i;
    for (i = 0; i < nvec; i++) {
        HVX_Vector va = *(const HVX_Vector *)(a + i * 128);
        *(HVX_Vector *)(out + i * 64) = Q6_Vh_vdmpy_VubRb(va, Rt);
    }
    for (int k = nvec * 32; k < g; k++) {
        int s0 = (int)a[4*k+0]*w[0] + (int)a[4*k+1]*w[1];
        int s1 = (int)a[4*k+2]*w[2] + (int)a[4*k+3]*w[3];
        out[2*k]   = (int16_t)s0;
        out[2*k+1] = (int16_t)s1;
    }
}
