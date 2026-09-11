/* EXPERT (achievability bar) — HVX vrmpy sum-of-squares.
 * Q6_Vw_vrmpy_VbVb(a,a) gives, per 32-bit lane, the sum of 4 signed a*a
 * products; accumulate across the array then horizontally reduce 32 lanes. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    const int vlen = sizeof(HVX_Vector); /* 128 */
    HVX_Vector acc = Q6_V_vzero();
    int i = 0;
    for (; i + vlen <= n; i += vlen) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        acc = Q6_Vw_vadd_VwVw(acc, Q6_Vw_vrmpy_VbVb(va, va));
    }
    int32_t tmp[32] __attribute__((aligned(128)));
    *(HVX_Vector *)tmp = acc;
    int32_t s = 0;
    for (int j = 0; j < 32; j++) s += tmp[j];
    for (; i < n; i++) s += (int32_t)a[i] * (int32_t)a[i];
    out[0] = s;
}
