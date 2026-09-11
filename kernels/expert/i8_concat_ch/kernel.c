/* sol_03: HVX unaligned copy â€” HVX_UVector for unaligned loads/stores, scalar tail */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static void hvx_memcpy_u(int8_t *dst, const int8_t *src, int len) {
    int i = 0;
    for (; i + 128 <= len; i += 128) {
        HVX_Vector v = *((HVX_UVector *)(src + i));
        *((HVX_UVector *)(dst + i)) = v;
    }
    for (; i < len; i++)
        dst[i] = src[i];
}

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out,
                      int C1, int C2, int H, int W) {
    int plane = H * W;
    hvx_memcpy_u(out, a, C1 * plane);
    hvx_memcpy_u(out + C1 * plane, b, C2 * plane);
}