/* 1D int8 array reverse -- the HVX expert.
 *
 * out[i] = in[n-1-i]. The whole job is a permutation, and HVX has an instruction for
 * exactly this shape: Q6_V_vdelta_VV routes lane i of its input to lane i ^ ctrl[i],
 * so a control vector with 0x7F in every byte sends lane i to lane i ^ 127 = 127 - i
 * and reverses all 128 lanes in one op.
 *
 * The blocks line up back-to-front: the destination block out[i .. i+127] is the
 * reverse of the source block ending at in[n-1-i], which starts at in[n-128-i]. n is
 * 501, so those source addresses are never 128B aligned even though `in` is -- hence
 * the two-load valign, the standard unaligned vector read. The second load can reach
 * up to 11 bytes past in[], but valign discards those lanes and the array is a static
 * in .bss with more statics after it, so nothing faults.
 *
 * 501 = 3*128 + 117, and the 117-element tail is copied scalar. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

static inline HVX_Vector load_ua(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    HVX_Vector v0 = *vp, v1 = *(vp + 1);
    return Q6_V_valign_VVR(v1, v0, (int)((uintptr_t)p & 127));
}

void candidate_kernel(const int8_t *in, int8_t *out, int n) {
    const HVX_Vector rev = Q6_V_vsplat_R(0x7F7F7F7F);   /* lane i -> lane 127 - i */
    int i = 0;
    for (; i + 128 <= n; i += 128)
        *(HVX_Vector *)(out + i) = Q6_V_vdelta_VV(load_ua(in + (n - 128 - i)), rev);
    for (; i < n; i++)
        out[i] = in[n - 1 - i];
}
