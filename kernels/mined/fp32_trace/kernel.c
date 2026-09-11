/* fp32_trace -- out0[0] = sum of x[i][i], the trace of a 480x480 fp32 matrix.
 *
 * Sources: the reference loop nest and the schedule in the prompt; the vendor
 * header <hvx_hexagon_protos.h>. No header from this repository.
 *
 * THE PAIR WITH fp32_diag_scale, AND WHAT IT ISOLATES
 * --------------------------------------------------
 * Same strided diagonal read (elements 1924 bytes apart, one cache line each, no
 * vector load can gather them), different consumer: that kernel scales and writes
 * 480 values, this one reduces them to a single scalar. Together they separate the
 * cost of the ADDRESSING from the cost of the arithmetic that follows, which is the
 * only way to say whether a diagonal kernel is bound by the gather or by the work.
 *
 * The answer for this one is: entirely by the gather. 480 scalar loads and then 15
 * vector adds plus a horizontal reduce -- the arithmetic is ~4% of the instructions
 * and no vectorisation of it can matter. A candidate that "optimised" the reduction
 * here would be optimising the part that is already free.
 *
 * ACCUMULATION ORDER differs from the reference's strict left-to-right, so the
 * result is tolerance-equal rather than bit-identical: the vector accumulator sums
 * 32 interleaved partial sums which are then folded pairwise. That is a tree, which
 * is more accurate than sequential, and this kernel's output is a single value whose
 * magnitude is ~sqrt(480) times a single element -- no cancellation structure to
 * worry about.
 *
 * ARITHMETIC: qf32 (IEEE fp32 add is unselectable on v75), converted to IEEE once at
 * the end.
 */
#include <stdint.h>
#include <stddef.h>          /* size_t, for the row offsets */
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#define N     480
#define COLS  480
#define LANES 32
#define NVEC  (N / LANES)              /* 15, exact */

#define VEC_ALIGN __attribute__((aligned(128)))

extern "C" void candidate_kernel(const float *v_x, float *out0) {
    /* Same staging as fp32_diag_scale: one scalar load per diagonal element, which
     * is the floor, into a vector-aligned buffer. */
    static float diag[N] VEC_ALIGN;
    for (int i = 0; i < N; i++)
        diag[i] = v_x[(size_t)i * COLS + i];

    const HVX_Vector *src = (const HVX_Vector *)diag;
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector acc = Q6_Vqf32_vadd_VsfVsf(zero, zero);      /* qf32 zero */
    for (int v = 0; v < NVEC; v++)
        acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, src[v]);

    /* Horizontal sum of 32 lanes: byte rotates of 64/32/16/8/4 for 4-byte lanes. */
    acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 64));
    acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 32));
    acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc, 16));
    acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc,  8));
    acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_vror_VR(acc,  4));

    static float lane[LANES] VEC_ALIGN;
    *(HVX_Vector *)lane = Q6_Vsf_equals_Vqf32(acc);
    out0[0] = lane[0];
}
