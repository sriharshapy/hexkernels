/* expert: same as solutions/s1.c (drill task, accelerable=false; 1.0x is fine). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, int16_t *out, int n) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        *(HVX_Vector *)(out + i) = Q6_Vh_vabs_Vh(va);
    }
    for (; i < n; i++) {
        int v = a[i];
        out[i] = (int16_t)(v < 0 ? -v : v);
    }
}
