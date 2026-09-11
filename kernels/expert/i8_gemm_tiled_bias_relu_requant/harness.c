/* i8_gemm_tiled_bias_relu_requant harness (v5, L3 composition spike).
 * Tiled int8 GEMM (M x K by K x N) + fused bias/ReLU/saturating-requant epilogue.
 * The harness owns main(): fills seeded inputs, computes the reference matmul via
 * HVX vrmpy (fast — a naive scalar M*N*K reference would blow the sim budget),
 * cross-checks that HVX reference against an INDEPENDENT scalar golden on the
 * first row (semantic guard: never trust one implementation), applies the scalar
 * epilogue, poisons output, times the candidate (kernel-only pcycles), and does a
 * bit-exact compare. Enables the HMX context AND an identity VTCM translation so a
 * candidate may compose HMX + DMA + VTCM.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef M
#define M 128
#endif
#ifndef N
#define N 128
#endif
#ifndef K
#define K 128
#endif

static uint8_t  A[M*K]   HVX_ALIGN;
static int8_t   B[K*N]   HVX_ALIGN;
static int8_t   Bt[N*K]  HVX_ALIGN;   /* B transposed for the vrmpy reference */
static int32_t  bias[N]  HVX_ALIGN;
static uint8_t  out[M*N] HVX_ALIGN;
static uint8_t  ref[M*N] HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}
static inline uint8_t saturate_u8(int v) {
    if (v > 255) v = 255;
    if (v <   0) v = 0;
    return (uint8_t)v;
}
/* Horizontal sum of the 32 word lanes of a vrmpy accumulator. */
static inline int32_t hreduce32(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

int main(void) {
    /* Map VTCM identity so DMA-capable candidates can stage crouton tiles on-chip. */
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x6E33u;
    /* A in 0..3 (positive: int8==uint8), B in -3..3. With K=128 the worst-case
     * |acc| = 128*3*3 = 1152 -> |acc*17/16| = 1224 < 2048, so the HMX 12-bit
     * requant field is exact (no saturation) and bit-exactness holds. */
    for (int i = 0; i < M*K; i++) A[i] = (uint8_t)(hvx_lcg(&s) % 4);           /* 0..3 */
    for (int i = 0; i < K*N; i++) B[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3); /* -3..3 */
    for (int j = 0; j < N;   j++) bias[j] = (int32_t)((int)(hvx_lcg(&s) % 4001) - 2000);
    bias[0] = -2000; bias[1] = 2000; bias[2] = 0; /* relu-zero, uint8-sat-high, passthrough */

    /* transpose B: Bt[j*K + k] = B[k*N + j] */
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) Bt[j*K + k] = B[k*N + j];

    /* HVX vrmpy reference matmul + scalar epilogue. K is a multiple of 128 so the
     * K-dim maps to whole vectors and no tail mask is needed. */
    const HVX_Vector zero = Q6_V_vzero();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            HVX_Vector vp = zero;
            for (int k = 0; k < K; k += 128) {
                HVX_Vector va = *(const HVX_UVector *)(A  + i*K + k);
                HVX_Vector vb = *(const HVX_UVector *)(Bt + j*K + k);
                vp = Q6_Vw_vrmpyacc_VwVbVb(vp, va, vb);
            }
            int acc = (int)hreduce32(vp);
            int r   = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[j];
            int relu   = biased > 0 ? biased : 0;
            ref[i*N + j] = saturate_u8(relu);
        }
    }

    /* Independent scalar golden on row 0 — guards the vrmpy reference's semantics. */
    for (int j = 0; j < N; j++) {
        int acc = 0;
        for (int k = 0; k < K; k++) acc += (int)A[0*K + k] * (int)B[k*N + j];
        int r = sx12((acc * 17 + 8) >> 4);
        int biased = r + bias[j];
        int relu = biased > 0 ? biased : 0;
        uint8_t g = saturate_u8(relu);
        if (g != ref[0*N + j]) {
            printf("HVXENV_REFCHECK_FAIL j=%d scalar=%d vrmpy=%d\n", j, (int)g, (int)ref[j]);
            return 2;
        }
    }

    for (int i = 0; i < M*N; i++) *((volatile unsigned char *)&out[i]) = 0xA5; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, bias, out, M, N, K); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < M*N; i++) {
        int g = (int)out[i], e = (int)ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = g; expv = e; } }
    }
    hvx_report(errors, M*N, fb, gotv, expv);
    return errors ? 1 : 0;
}
