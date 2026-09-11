/* Near-miss: computes MIN instead of MAX (operator confusion). Wrong
 * whenever the data isn't uniform (guaranteed by the harness's randomized
 * input + planted max/min values). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *a, int n, uint8_t *out) {
    const int vlen = 128;
    HVX_Vector acc = Q6_V_vsplat_R(0xFFFFFFFFu);
    int i = 0;
    for (; i + vlen <= n; i += vlen)
        acc = Q6_Vub_vmin_VubVub(acc, *(const HVX_Vector *)(a + i));

    uint8_t lanes[128] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes = acc;
    uint8_t m = 0xFF;
    for (int j = 0; j < 128; j++) if (lanes[j] < m) m = lanes[j];
    for (; i < n; i++) if (a[i] < m) m = a[i];
    out[0] = m;
}
