/* HMX tile-matmul helper primitives (Task 1 of the helper-API-training design,
 * docs/superpowers/specs/2026-07-22-hmx-helper-api-training-design.md).
 *
 * Header-only, sim-verified (run_artifacts/v6/hmx_helper/verify_helper.c).
 * Generalizes the proven 32x32-crouton HMX matmul flow (lifted verbatim from
 * datasets/v6/tasks/i8_matmul_bias_relu_hmx/expert.c and the fp16 siblings
 * fp16_matmul_hmx_32x32 / hmx_matmul_fp16_deepk) to arbitrary M/N/K tile grids
 * (M,N,K all multiples of 32). Caller does epilogue (bias/relu/requant/cast)
 * in HVX/scalar -- this primitive writes the raw (post-requant, for int8)
 * matmul result straight to the output buffer.
 *
 * NOTE on placement: the design doc (2026-07-22) says this belongs in
 * evaluate.py's COMMON_DIR so future task solutions can `#include
 * "hmx_helpers.h"` with zero compile-harness changes. evaluate.py's actual
 * COMMON_DIR is `m1_driver/tasks/common/` (see evaluate.py:28), NOT
 * `m1_driver/harness/` as the task brief says (that path is stale --
 * harness_common.h also isn't really there). This file is intentionally
 * duplicated verbatim at BOTH `m1_driver/harness/hmx_helpers.h` (brief's
 * literal instruction) and `m1_driver/tasks/common/hmx_helpers.h` (the real
 * COMMON_DIR, so the design's "zero-compile-harness-changes" claim actually
 * holds for Task 2). See task-1-report.md for the full note. */
#ifndef HMX_HELPERS_H
#define HMX_HELPERS_H

/* R&D ONLY -- NOT FOR FORGE V2.
 *
 * The tile-matmul helpers below were lifted from expert solutions, so they carry
 * that provenance. Forge v2's claim is that its kernels derive from the generated
 * reference, the schedule annotation and the VENDOR intrinsic headers only; a
 * forge v2 kernel built on this header has a different and weaker chain than the
 * one that pipeline states, which is a provenance failure however correct the
 * kernel is.
 *
 * This header sits in the directory forge2's verifier passes to `-I`, so the
 * include was silently available. `verify.py` now defines FORGE2_BUILD on every
 * forge v2 compile, making that a hard error with a reason instead. The source
 * scan in `forge2.provenance.rd_leaks` is the other half -- it catches a COPY of
 * the body, which no include guard can. */
#ifdef FORGE2_BUILD
#error "hmx_helpers.h is R&D-only and must not be used by a forge v2 kernel. Derive from the reference, the schedule, and <hmx_hexagon_protos.h>. See hexbench/forge2/provenance.py."
#endif
#include <stdint.h>
#include "harness_common.h"   /* hvx_hmx_i8_{act,wgt,out}_off, HVX_VTCM_BASE, HVX_ALIGN */
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>

/* Sign-extend the HMX 0x40-config 12-bit requant field: r = (acc*17+8)>>4,
 * folded to a two's-complement 12-bit value. This is NOT a full int32
 * accumulate -- large K/large inputs wrap (see design doc "semantics
 * caveat"). Exported (not just internal) so callers/tests can reuse the exact
 * documented post-processing formula. */
static inline int _hmx_sx12(int f) {
    int v = f & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

/* HMX int8 tile matmul, pure hardware atom (bit-exact vs the documented
 * semantics): field[i][j] = (sum_{k<K} A[i*K+k]*B[k*N+j]) * 17 + 8 >> 4,
 * masked to the raw 12-bit two's-complement HMX requant field (NOT
 * sign-extended -- see _hmx_sx12 above for that post-step).
 * A uint8 row-major MxK, B int8 row-major KxN, field uint16 row-major MxN.
 * M,N,K must be multiples of 32 (crouton tile edge). Loops over (M/32 x N/32)
 * output tiles, accumulating each over K/32 K-tiles via repeated
 * (activation,weight) mxmem loads into ONE HMX accumulator per output tile
 * (mxclracc once per tile, before its K-loop) -- identical mechanism to the
 * i8_matmul_bias_relu_hmx / i8_matmul_hmx_96x96 / hmx_matmul_i8_rect_32x64
 * experts, generalized to arbitrary M/N/K instead of a fixed 2x2 or 3x3 grid.
 * Crouton pack/unpack is staged through cacheable buffers and moved to/from
 * VTCM in bulk 128B vector copies (scalar VTCM access is ~48 cyc/access in
 * timing mode). This writes the raw crouton readback straight to field;
 * callers needing the old sx12'd int32 semantics should use the
 * hmx_tile_matmul_i8 wrapper below (or apply _hmx_sx12 themselves). */
static inline void hmx_tile_matmul_i8_field(const uint8_t *A, const int8_t *B,
                                             uint16_t *field, int M, int N, int K) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u); /* 2048B */
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u); /* 1024B */
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;   /* cacheable crouton staging */
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;    /* even/low bytes stay 0 */
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40; /* HMX requant config: scale 17/16, bias 0 */

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;                       /* crouton tile edge */
    const int mt = M / T, nt = N / T, kt_n = K / T;

    for (int ti = 0; ti < mt; ti++) {
        for (int tj = 0; tj < nt; tj++) {
            __asm__ volatile("mxclracc\n");             /* clear accumulator per out tile */
            for (int kt = 0; kt < kt_n; kt++) {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = A[(ti*T + i)*K + (kt*T + k)];
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = B[(kt*T + k)*N + (tj*T + j)];
                { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
                  for (int b = 0; b < 16; b++) d[b] = s[b]; }
                { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
                  for (int b = 0; b < 8; b++) d[b] = s[b]; }
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++)
                    field[(ti*T + i)*N + (tj*T + j)] = aOut[hvx_hmx_i8_out_off(i, j)];
        }
    }
}

/* HMX int8 tile matmul, thin convenience wrapper (bit-exact vs the
 * documented semantics): C[i][j] = sx12( (sum_{k<K} A[i*K+k]*B[k*N+j]) * 17
 * + 8 >> 4 ). A uint8 row-major MxK, B int8 row-major KxN, C int32 row-major
 * MxN. Calls the hmx_tile_matmul_i8_field atom above into a static scratch
 * then applies the documented _hmx_sx12 post-step per element -- this is
 * exactly the old (pre-split) body's behavior. The scratch is sized for the
 * largest output grid in the benchmark (128x128); for a larger M*N, call the
 * _field atom directly with your own field buffer. Caller does
 * bias/relu/requant/cast beyond that; this writes the sx12'd int32 to C. */
static inline void hmx_tile_matmul_i8(const uint8_t *A, const int8_t *B,
                                      int32_t *C, int M, int N, int K) {
    static uint16_t _fld_i8[128*128] HVX_ALIGN;
    hmx_tile_matmul_i8_field(A, B, _fld_i8, M, N, K);
    for (int t = 0; t < M*N; t++) C[t] = _hmx_sx12((int)_fld_i8[t]);
}

/* HMX fp16 tile matmul, pure hardware atom: field[i][j] = sum_k
 * A[i*K+k]*B[k*N+j], fp16 in / fp16 out (raw crouton readback, no float
 * upcast). M,N,K multiples of 32. Same M/N/K-tile generalization as the int8
 * primitive above, but using the fp16 crouton layout/intrinsics
 * (off(r,c)=(r/2)*64+c*2+(r&1), Q6_mxclracc_hf /
 * Q6_{activation,weight}_hf_mxmem_RR / Q6_mxmem_AR_after_hf) lifted from
 * fp16_matmul_hmx_32x32/expert.c (single tile) and
 * hmx_matmul_fp16_deepk/expert.c (K-accumulation: one mxclracc per output
 * tile, before its K-loop, matching the int8 accumulator-reuse pattern).
 *
 * PRECISION -- CORRECTED 2026-08-03, the previous note here was backwards and
 * it cost a mechanism (a forge v2 author cited it as a reason to decline HMX).
 * It said the accumulator "is fp16-precision internally" and that deep-K
 * "needs a wider tolerance". What is fp16 is the STORE (":after.hf=acc" -- there
 * is no fp32-accumulate store variant), not the accumulation. MEASURED: at
 * K=128 in ONE accumulator the result is BIT-EXACT against an fp32
 * accumulation rounded once to fp16, and the split-K structure the old wording
 * implies is strictly WORSE (106/1024 outside tolerance). So keep one
 * mxclracc per OUTPUT TILE, as this atom does, and depth costs you nothing.
 * Tolerance comparison is still required -- HVX/HMX float arithmetic is
 * non-IEEE qf16 -- but it does not need widening with K.
 *
 * Callers needing a float32 interface should use the hmx_tile_matmul_fp16
 * wrapper below (or upcast themselves). */
static inline void hmx_tile_matmul_fp16_field(const __fp16 *A, const __fp16 *B,
                                               __fp16 *field, int M, int N, int K) {
    __fp16 *vA = (__fp16 *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    __fp16 *vB = (__fp16 *)(uintptr_t)(HVX_VTCM_BASE + 0x0800u);  /* 2048B apart */
    __fp16 *vO = (__fp16 *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    /* SCALE. Without this the read-out is at ZERO scale and every element stores
     * as exactly 0.0 -- see the note above `hmx_tile_matmul_fp16` for the
     * measurement. 0x3C00 is fp16 1.0 in the low halfword (scale); the high
     * halfword is the offset, 0. */
    volatile uint32_t *vBias = (volatile uint32_t *)(uintptr_t)(HVX_VTCM_BASE + 0x1800u);

    static __fp16 aA[32*32] HVX_ALIGN, aB[32*32] HVX_ALIGN, aO[32*32] HVX_ALIGN;

    for (int i = 0; i < 512; i++) vBias[i] = 0x00003C00u;

    const int T = 32;
    const int mt = M / T, nt = N / T, kt_n = K / T;

    for (int ti = 0; ti < mt; ti++) {
        for (int tj = 0; tj < nt; tj++) {
            Q6_mxclracc_hf();                       /* clear accumulator once per out tile */
            for (int kt = 0; kt < kt_n; kt++) {
                for (int r = 0; r < T; r++)
                    for (int c = 0; c < T; c++)
                        aA[hvx_crouton_off(r, c)] = A[(ti*T + r)*K + (kt*T + c)];
                for (int r = 0; r < T; r++)
                    for (int c = 0; c < T; c++)
                        aB[hvx_crouton_off(r, c)] = B[(kt*T + r)*N + (tj*T + c)];
                { HVX_Vector *s = (HVX_Vector *)aA, *d = (HVX_Vector *)vA;
                  for (int b = 0; b < 16; b++) d[b] = s[b]; }
                { HVX_Vector *s = (HVX_Vector *)aB, *d = (HVX_Vector *)vB;
                  for (int b = 0; b < 16; b++) d[b] = s[b]; }
                Q6_activation_hf_mxmem_RR((unsigned int)(uintptr_t)vA, 2047);
                Q6_weight_hf_mxmem_RR((unsigned int)(uintptr_t)vB, 2047);
            }
            Q6_bias_mxmem_A((void *)(uintptr_t)vBias);
            Q6_mxmem_AR_after_hf(vO, 2047);
            __asm__ volatile("isync\n\t");
            { HVX_Vector *s = (HVX_Vector *)vO, *d = (HVX_Vector *)aO;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            for (int r = 0; r < T; r++)
                for (int c = 0; c < T; c++)
                    field[(ti*T + r)*N + (tj*T + c)] = aO[hvx_crouton_off(r, c)];
        }
    }
}

/* HMX fp16 tile matmul, thin convenience wrapper: C[i][j] = sum_k
 * A[i*K+k]*B[k*N+j], fp16 in / fp32 out. Calls the hmx_tile_matmul_fp16_field
 * atom above into a static scratch (sized for the largest supported grid,
 * 96x96) then upcasts each element to float -- this is exactly the old
 * (pre-split) body's behavior, unchanged: the HMX float accumulator/store is
 * fp16-precision internally, so "fp32 out" describes the interface type,
 * not extra hardware precision. */
static inline void hmx_tile_matmul_fp16(const __fp16 *A, const __fp16 *B,
                                        float *C, int M, int N, int K) {
    static __fp16 _fld_f16[128*128] HVX_ALIGN;
    hmx_tile_matmul_fp16_field(A, B, _fld_f16, M, N, K);
    for (int t = 0; t < M*N; t++) C[t] = (float)_fld_f16[t];
}

#endif
