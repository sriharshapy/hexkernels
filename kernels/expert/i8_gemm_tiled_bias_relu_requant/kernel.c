/* EXPERT (achievability bar) — the L3 composition: HMX + DMA + VTCM double-buffer
 * + HVX epilogue.  Tiled int8 GEMM (M x K by K x N) on the HMX matrix engine.
 *
 * COMPOSITION (the thing this spike proves — HMX and uDMA coexisting in VTCM):
 *   - VTCM is PARTITIONED into disjoint regions: two double-buffered crouton
 *     operand slots (activation 2KB x2, weight 1KB x2), a requant-config tile, and
 *     an output crouton tile. HMX reads its operands from VTCM via mxmem; uDMA
 *     writes the NEXT K-tile's crouton into the *alternate* VTCM slot at the same
 *     time. Different regions -> no fault, no aliasing.
 *   - DOUBLE-BUFFER: for each output tile we pack the crouton for K-tile kt in
 *     cacheable DDR (cheap), then uDMA it DDR->VTCM. While HMX computes K-tile kt
 *     from VTCM slot (kt&1), the uDMA for kt+1 streams into slot (kt+1)&1. The DMA
 *     latency hides behind HMX compute.
 *   - EPILOGUE: the HMX 0x40-config store yields the requant field r; an HVX/scalar
 *     epilogue then adds per-column bias, ReLUs, and saturates to uint8.
 *
 * Crouton pack/unpack is staged through cacheable buffers (scalar VTCM access is
 * ~48 cyc in timing mode); the DDR->VTCM move is the uDMA. HMX gives the win over
 * the HVX vrmpy baseline; DMA+VTCM provide the on-chip streaming that makes this
 * the canonical on-device inference kernel.
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md +
 *         datasets/v5/tasks/i8_vadd_dma/expert.c (uDMA double-buffer). */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

/* uDMA Type-0 (1D) descriptor, 16 bytes. MUST be static/global (a stack
 * descriptor silently no-ops at -O2). */
typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
static desc_t d_a, d_w;   /* chained: activation tile -> weight tile */

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

#define ACT_BYTES 2048u
#define WGT_BYTES 1024u

void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias,
                      uint8_t *out, int M, int N, int K) {
    /* --- VTCM partition (disjoint regions; HMX operands + DMA targets coexist) --- */
    const uint32_t vAct0 = HVX_VTCM_BASE + 0x0000u;  /* 2KB */
    const uint32_t vAct1 = HVX_VTCM_BASE + 0x0800u;  /* 2KB */
    const uint32_t vWgt0 = HVX_VTCM_BASE + 0x1000u;  /* 1KB */
    const uint32_t vWgt1 = HVX_VTCM_BASE + 0x1400u;  /* 1KB */
    const uint32_t vBias = HVX_VTCM_BASE + 0x2000u;  /* 2KB requant config */
    const uint32_t vOut  = HVX_VTCM_BASE + 0x3000u;  /* 2KB output crouton */

    /* Cacheable crouton staging, double-buffered so a prefetch's source is stable
     * while the current tile is consumed. */
    static uint8_t aAct[2][ACT_BYTES] HVX_ALIGN;
    static int8_t  aWgt[2][WGT_BYTES] HVX_ALIGN;
    static uint16_t aOut[32*32]       HVX_ALIGN;
    static uint8_t  aBias[ACT_BYTES]  HVX_ALIGN;

    for (int p = 0; p < 2; p++) for (int i = 0; i < (int)ACT_BYTES; i++) aAct[p][i] = 0; /* dead low bytes */
    for (int i = 0; i < (int)ACT_BYTES; i++) aBias[i] = 0x40;   /* requant: scale 17/16, bias 0 */
    { HVX_Vector *sb = (HVX_Vector *)aBias, *db = (HVX_Vector *)(uintptr_t)vBias;
      for (int b = 0; b < 16; b++) db[b] = sb[b]; }             /* bulk-copy config -> VTCM once */

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;
    const int ntM = M / T, ntN = N / T, nkt = K / T;

    for (int ti = 0; ti < ntM; ti++) {
        for (int tj = 0; tj < ntN; tj++) {

            /* --- pack + DMA K-tile 0 into VTCM slot 0 (prologue) --- */
            {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[0][hvx_hmx_i8_act_off(i, k)] = A[(ti*T + i)*K + (0*T + k)];
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        aWgt[0][hvx_hmx_i8_wgt_off(k, j)] = B[(0*T + k)*N + (tj*T + j)];
                d_a.next = (uint32_t)(uintptr_t)&d_w; d_a.ctrl = ACT_BYTES;
                d_a.src = (uint32_t)(uintptr_t)aAct[0]; d_a.dst = vAct0;
                d_w.next = 0; d_w.ctrl = WGT_BYTES;
                d_w.src = (uint32_t)(uintptr_t)aWgt[0]; d_w.dst = vWgt0;
                Q6_dmstart_A(&d_a); Q6_R_dmwait();
            }

            __asm__ volatile("mxclracc\n");   /* clear HMX accumulator once per output tile */

            for (int kt = 0; kt < nkt; kt++) {
                int cur = kt & 1;
                uint32_t vAct_cur = cur ? vAct1 : vAct0;
                uint32_t vWgt_cur = cur ? vWgt1 : vWgt0;

                /* Prefetch the NEXT K-tile: pack in cacheable DDR, DMA into the
                 * ALTERNATE VTCM slot (async) while HMX computes the current one. */
                if (kt + 1 < nkt) {
                    int nb = (kt + 1) & 1;
                    for (int i = 0; i < T; i++)
                        for (int k = 0; k < T; k++)
                            aAct[nb][hvx_hmx_i8_act_off(i, k)] = A[(ti*T + i)*K + ((kt+1)*T + k)];
                    for (int k = 0; k < T; k++)
                        for (int j = 0; j < T; j++)
                            aWgt[nb][hvx_hmx_i8_wgt_off(k, j)] = B[((kt+1)*T + k)*N + (tj*T + j)];
                    d_a.next = (uint32_t)(uintptr_t)&d_w; d_a.ctrl = ACT_BYTES;
                    d_a.src = (uint32_t)(uintptr_t)aAct[nb]; d_a.dst = nb ? vAct1 : vAct0;
                    d_w.next = 0; d_w.ctrl = WGT_BYTES;
                    d_w.src = (uint32_t)(uintptr_t)aWgt[nb]; d_w.dst = nb ? vWgt1 : vWgt0;
                    Q6_dmstart_A(&d_a);   /* async — overlaps the HMX matmul below */
                }

                /* one HMX matmul; accumulates into the (uncleared) accumulator */
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct_cur), "r"(lim_a), "r"(vWgt_cur), "r"(lim_w) : "memory");

                if (kt + 1 < nkt) Q6_R_dmwait();   /* finish the prefetch before reusing the slot */
            }

            /* HMX requant store -> VTCM output crouton */
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");

            /* bulk-copy VTCM crouton -> cacheable, then the fused HVX epilogue.
             * The crouton output layout is off(r,c) = (r/2)*64 + c*2 + (r&1), so ONE
             * 128B vector is exactly one row PAIR: 32 columns of the even row
             * interleaved with 32 of the odd. vshuffe/vshuffo split it into the two
             * rows -- both land in the LOW 32 lanes, which is why only Q6_V_lo_W of
             * the widening pair is taken (a vdeal + lo/hi version failed exactly half
             * the elements, the signature of a wrong upper-half ordering assumption).
             * The vasl/vasr pair sign-extends the 12-bit requant field, vsxt widens to
             * int32, and then bias -> ReLU -> saturating narrow all run in vector
             * lanes: vadd for the per-column bias (bias[] is HVX_ALIGN and tj*T is a
             * multiple of 32 words, so that load is aligned), vmax against zero for the
             * ReLU, and the vpack:sat chain word -> halfword -> UNSIGNED byte for the
             * requant, which is where the [0,255] clamp happens in hardware. Saturating
             * twice equals clamping once because saturation is monotone. Each vpack
             * takes the same vector as both operands, so the low 32 bytes hold this row
             * whichever operand fills the low half. The hand-off to out[] is scalar
             * because a 32-column uint8 tile row is 32 B against an N-byte row stride,
             * so no aligned vector store covers it. */
            { HVX_Vector *s = (HVX_Vector *)(uintptr_t)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            {
                const HVX_Vector vzero = Q6_V_vzero();
                const HVX_Vector vb    = *(const HVX_Vector *)(bias + tj*T);
                static uint8_t aPack[128] HVX_ALIGN;
                for (int rp = 0; rp < T/2; rp++) {
                    HVX_Vector fv = ((const HVX_Vector *)aOut)[rp];
                    HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(fv, fv), 4), 4);
                    HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(fv, fv), 4), 4);
                    HVX_Vector w[2];
                    w[0] = Q6_Vw_vmax_VwVw(Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(se)), vb), vzero);
                    w[1] = Q6_Vw_vmax_VwVw(Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(so)), vb), vzero);
                    for (int h = 0; h < 2; h++) {
                        HVX_Vector hw = Q6_Vh_vpack_VwVw_sat(w[h], w[h]);
                        *(HVX_Vector *)aPack = Q6_Vub_vpack_VhVh_sat(hw, hw);
                        uint8_t *dst = out + (ti*T + 2*rp + h)*N + tj*T;
                        for (int j = 0; j < T; j++) dst[j] = aPack[j];
                    }
                }
            }
        }
    }
}
